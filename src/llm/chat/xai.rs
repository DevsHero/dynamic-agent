use async_trait::async_trait;
use std::error::Error as StdError;
use std::pin::Pin;
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use serde::{Deserialize, Serialize};
use log::info;
use reqwest::Client as HttpClient;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, AUTHORIZATION};

use super::{ChatClient, CompletionResponse };
use crate::llm::LlmConfig;
use rllm::builder::LLMBackend;

#[derive(Debug)]
pub struct XAIChatClient {
    http: HttpClient,
    api_key: String,
    model: String,
    base_url: Option<String>,
}

#[derive(Serialize)]
struct XAIRequest {
    model: String,
    messages: Vec<XAIMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize, Deserialize)]
struct XAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct XAIStreamResponse {
    choices: Vec<XAIChoice>,
}

#[derive(Deserialize, Debug)]
struct XAIChoice {
    delta: XAIDelta,
}

#[derive(Deserialize, Debug)]
struct XAIDelta {
    content: Option<String>,
}

impl XAIChatClient {
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let chat_model = model.unwrap_or_else(|| "grok-3-latest".to_string());
        
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        
        let http_client = HttpClient::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| Box::new(e) as Box<dyn StdError + Send + Sync>)?;

        Ok(Self { 
            http: http_client,
            api_key,
            model: chat_model,
            base_url
        })
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let api_key = config.api_key
            .clone()
            .ok_or_else(|| "XAI API key is required".to_string())?;
        
        let model = config.completion_model.clone();
        let base_url = config.base_url.clone();

        Self::new(api_key, model, base_url)
    }
    
    async fn generate_stream(
        &self,
        prompt: &str
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>> {
        let url = self.base_url.clone().unwrap_or_else(|| "https://api.x.ai/v1/chat/completions".to_string());
        
        let messages = vec![XAIMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        
        let req = XAIRequest {
            model: self.model.clone(),
            messages,
            stream: true,
            temperature: Some(0.7), 
        };
        
        let (tx, rx) = mpsc::channel(32);
        
        let client = self.http.clone();
        let auth_header = format!("Bearer {}", self.api_key);
        
        tokio::spawn(async move {
            let mut builder = client.post(&url).json(&req);
            builder = builder.header(AUTHORIZATION, auth_header);
            
            match builder.send().await {
                Ok(resp) => {
                    if let Err(e) = resp.error_for_status_ref() {
                        info!("XAI API error: {}", e);
                        let _ = tx.send(Err(Box::new(e) as _)).await;
                        return;
                    }
                    
                    let mut stream = resp.bytes_stream();
                    
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                if let Ok(text) = String::from_utf8(chunk.to_vec()) {
                                    info!("XAI raw chunk: {}", text);
                                    
                                    for line in text.lines() {
                                        if line.is_empty() || line == "data: [DONE]" {
                                            continue;
                                        }
                                        
                                        if let Some(data) = line.strip_prefix("data: ") {
                                            match serde_json::from_str::<XAIStreamResponse>(data) {
                                                Ok(stream_resp) => {
                                                    for choice in stream_resp.choices {
                                                        if let Some(content) = choice.delta.content {
                                                            if !content.is_empty() {
                                                                if tx.send(Ok(content)).await.is_err() {
                                                                    return;
                                                                }
                                                            }
                                                        }
                                                    }
                                                },
                                                Err(e) => {
                                                    info!("JSON parse error: {} for data: {}", e, data);
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            Err(e) => {
                                let _ = tx.send(Err(Box::new(e) as _)).await;
                                return;
                            }
                        }
                    }
                },
                Err(e) => {
                    let _ = tx.send(Err(Box::new(e) as _)).await;
                }
            }
        });
        
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

#[async_trait]
impl ChatClient for XAIChatClient {
    async fn complete(
        &self,
        prompt: &str
    ) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>> {
        let url = self.base_url.clone().unwrap_or_else(|| "https://api.x.ai/v1/chat/completions".to_string());
        
        let messages = vec![XAIMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        
        let req = XAIRequest {
            model: self.model.clone(),
            messages,
            stream: false,
            temperature: Some(0.7),
        };
        
        let client = self.http.clone();
        let auth_header = format!("Bearer {}", self.api_key);
        
        let resp = client.post(&url)
            .header(AUTHORIZATION, auth_header)
            .json(&req)
            .send()
            .await?
            .error_for_status()?;
        
        #[derive(Deserialize)]
        struct XAIResponse {
            choices: Vec<XAIResponseChoice>,
        }
        
        #[derive(Deserialize)]
        struct XAIResponseChoice {
            message: XAIMessage,
        }
        
        let xai_resp = resp.json::<XAIResponse>().await?;
        let content = xai_resp.choices.first()
            .ok_or_else(|| "No response from XAI API".to_string())?
            .message.content.clone();
        
        Ok(CompletionResponse { response: content })
    }
    
    async fn stream_completion(
        &self,
        prompt: &str
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>> {
        self.generate_stream(prompt).await
    }
    
    fn supports_native_streaming(&self) -> bool {
        true
    }
    
    fn get_api_key(&self) -> String {
        self.api_key.clone()
    }
    
    fn get_model(&self) -> String {
        self.model.clone()
    }
    
    fn get_base_url(&self) -> Option<String> {
        self.base_url.clone()
    }
    
    fn get_llm_backend(&self) -> LLMBackend {
        LLMBackend::XAI
    }
}
