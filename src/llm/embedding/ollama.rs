use async_trait::async_trait;
use std::error::Error as StdError;
use super::{ EmbeddingClient, EmbeddingResponse };
use super::super::LlmConfig;
use rllm::{ builder::{ LLMBackend, LLMBuilder }, LLMProvider };

pub struct OllamaEmbeddingClient {
    llm: Box<dyn LLMProvider + Send + Sync>,
}

impl OllamaEmbeddingClient {
    pub fn new(
        base_url: Option<String>,
        model: Option<String>
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let url = base_url.unwrap_or_else(|| "http://localhost:11434".to_string());
        let embed_model = model.unwrap_or_else(|| "nomic-embed-text".to_string());

        let builder = LLMBuilder::new()
            .backend(LLMBackend::Ollama)
            .base_url(url)
            .model(embed_model)
            .stream(false);

        let llm_provider = builder.build()?;

        Ok(Self {
            llm: llm_provider,
        })
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let model = config.embedding_model.clone();
        Self::new(config.base_url.clone(), model)
    }
}

#[async_trait]
impl EmbeddingClient for OllamaEmbeddingClient {
    async fn embed(
        &self,
        text: &str
    ) -> Result<EmbeddingResponse, Box<dyn StdError + Send + Sync>> {
        let mut embeddings = self.llm.embed(vec![text.to_string()]).await?;
        let embedding = embeddings
            .pop()
            .ok_or_else(|| "Ollama embedding generation returned no results".to_string())?;

        Ok(EmbeddingResponse { embedding })
    }
}
