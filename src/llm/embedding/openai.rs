use async_trait::async_trait;
use rllm::{ builder::{ LLMBackend, LLMBuilder }, LLMProvider };
use std::error::Error as StdError;
use super::super::LlmConfig;
use super::{ EmbeddingClient, EmbeddingResponse };

pub struct OpenAIEmbeddingClient {
    llm: Box<dyn LLMProvider + Send + Sync>,
}

impl OpenAIEmbeddingClient {
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
        dimensions: Option<u32>
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let model_name = model.unwrap_or_else(|| "text-embedding-3-small".to_string());

        let mut builder = LLMBuilder::new()
            .backend(LLMBackend::OpenAI)
            .api_key(api_key)
            .model(&model_name);

        if let Some(url) = base_url {
            builder = builder.base_url(url);
        }
        if let Some(dims) = dimensions {
            builder = builder.embedding_dimensions(dims);
        }

        let llm = builder.build()?;

        Ok(Self { llm })
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let api_key = config.api_key
            .clone()
            .ok_or_else(|| "OpenAI API key is required for OpenAIEmbeddingClient".to_string())?;
        let model = config.embedding_model.clone();
        let base_url = config.base_url.clone();
        let dimensions = None;

        Self::new(api_key, model, base_url, dimensions)
    }
}

#[async_trait]
impl EmbeddingClient for OpenAIEmbeddingClient {
    async fn embed(
        &self,
        text: &str
    ) -> Result<EmbeddingResponse, Box<dyn StdError + Send + Sync>> {
        let mut embeddings = self.llm.embed(vec![text.to_string()]).await?;
        let embedding = embeddings
            .pop()
            .ok_or_else(|| "OpenAI embedding generation returned no results".to_string())?;

        Ok(EmbeddingResponse { embedding })
    }
}
