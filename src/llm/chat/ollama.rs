use reqwest::Client as HttpClient;
use serde::{ Deserialize, Serialize };
use std::error::Error;
use async_trait::async_trait;
use std::error::Error as StdError;
use super::{ ChatClient, CompletionResponse };
use crate::llm::LlmConfig;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio_stream::wrappers::ReceiverStream;
use tokio::sync::mpsc;
use log::info;
use rllm::builder::LLMBackend;

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

#[derive(Deserialize)]
struct StreamResponse {
    response: String,
    done: bool,
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
    
    pub async fn generate_stream(
        &self,
        prompt: &str
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>> {
        let url = format!("{}/api/generate", self.base_url);
        let req = GenerateRequest {
            model: self.completion_model.clone(),
            prompt: prompt.to_string(),
            stream: true, 
        };
        
        let (tx, rx) = mpsc::channel(32);
        let client = self.http.clone();

        tokio::spawn(async move {
            match client.post(&url).json(&req).send().await {
                Ok(response) => {
                    if !response.status().is_success() {
                        let err_msg = format!("HTTP error: {}", response.status());
                        let _ = tx.send(Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, err_msg)) as _)).await;
                        return;
                    }
                    let mut stream = response.bytes_stream();
                    
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                if let Ok(text) = String::from_utf8(chunk.to_vec()) {
                                    
                                    for line in text.lines() {
                                        if line.is_empty() {
                                            continue;
                                        }
                                        
                                        match serde_json::from_str::<StreamResponse>(line) {
                                            Ok(stream_resp) => {
                                                if !stream_resp.response.is_empty() {
                                                    if tx.send(Ok(stream_resp.response)).await.is_err() {
                                                        break;
                                                    }
                                                }
                                                
                                                if stream_resp.done {
                                                    break;
                                                }
                                            },
                                            Err(e) => {
                                                info!("JSON parse error: {} for line: {}", e, line);
                                                continue; 
                                            }
                                        }
                                    }
                                }
                            },
                            Err(e) => {
                                let _ = tx.send(Err(Box::new(e) as Box<dyn StdError + Send + Sync>)).await;
                                break;
                            }
                        }
                    }
                },
                Err(e) => {
                    let _ = tx.send(Err(Box::new(e) as Box<dyn StdError + Send + Sync>)).await;
                }
            }
        });
        
        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
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
    
    async fn stream_completion(
        &self,
        prompt: &str
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>> {
        self.generate_stream(prompt).await
    }

    fn get_api_key(&self) -> String {
        "".to_string()
    }

    fn get_model(&self) -> String {
        self.completion_model.clone()
    }

    fn get_base_url(&self) -> Option<String> {
        Some(self.base_url.clone())
    }

    fn get_llm_backend(&self) -> LLMBackend {
        LLMBackend::Ollama
    }

    fn supports_native_streaming(&self) -> bool {
        true
    }
}
