use async_trait::async_trait;
use std::error::Error as StdError;
use super::{ EmbeddingClient, EmbeddingResponse };
use super::super::LlmConfig;
use rllm::{ builder::{ LLMBackend, LLMBuilder }, LLMProvider };

pub struct GoogleEmbeddingClient {
    llm: Box<dyn LLMProvider + Send + Sync>,
}

impl GoogleEmbeddingClient {
    pub fn new(
        api_key: String,
        model: Option<String>
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let embed_model = model.unwrap_or_else(|| "text-embedding-004".to_string());

        let builder = LLMBuilder::new()
            .backend(LLMBackend::Google)
            .api_key(api_key)
            .model(embed_model)
            .stream(false);

        let llm_provider = builder.build()?;

        Ok(Self {
            llm: llm_provider,
        })
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let api_key = config.api_key
            .clone()
            .ok_or_else(|| "Google API key is required for GoogleEmbeddingClient".to_string())?;
        let model = config.embedding_model.clone();
        Self::new(api_key, model)
    }
}

#[async_trait]
impl EmbeddingClient for GoogleEmbeddingClient {
    async fn embed(
        &self,
        text: &str
    ) -> Result<EmbeddingResponse, Box<dyn StdError + Send + Sync>> {
        let mut embeddings = self.llm.embed(vec![text.to_string()]).await?;
        let embedding = embeddings
            .pop()
            .ok_or_else(|| "Google embedding generation returned no results".to_string())?;

        Ok(EmbeddingResponse { embedding })
    }
}
