use reqwest::Client as HttpClient;
use serde::{ Deserialize, Serialize };
use std::error::Error;
use async_trait::async_trait;
use std::error::Error as StdError;
use super::{ ChatClient, CompletionResponse };
use crate::llm::LlmConfig;

#[derive(Debug)]
pub struct OllamaClient {
    http: HttpClient,
    base_url: String,
    completion_model: String,
}

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
pub struct GenerateResponse {
    pub response: String,
}

impl OllamaClient {
    pub fn new(base_url: Option<String>, completion_model: Option<String>) -> Self {
        let model = completion_model.unwrap_or_else(|| "cogito:3b".to_string());
        let url = base_url.unwrap_or_else(|| "http://localhost:11434".into());

        Self {
            http: HttpClient::new(),
            base_url: url,
            completion_model: model,
        }
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        if config.llm_type != crate::llm::LlmType::Ollama {
            return Err("Invalid config type for OllamaClient".into());
        }

        Ok(Self::new(config.base_url.clone(), config.completion_model.clone()))
    }

    pub async fn generate(
        &self,
        prompt: &str
    ) -> Result<GenerateResponse, Box<dyn Error + Send + Sync>> {
        let url = format!("{}/api/generate", self.base_url);
        let req = GenerateRequest {
            model: self.completion_model.clone(),
            prompt: prompt.to_string(),
            stream: false,
        };
        let resp = self.http.post(&url).json(&req).send().await?.error_for_status()?;
        let data = resp.json::<GenerateResponse>().await?;
        Ok(data)
    }
}

#[async_trait]
impl ChatClient for OllamaClient {
    async fn complete(
        &self,
        prompt: &str
    ) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>> {
        let gen_resp = self.generate(prompt).await?;

        Ok(CompletionResponse { response: gen_resp.response })
    }
}
