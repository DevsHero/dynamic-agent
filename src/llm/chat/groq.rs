use async_trait::async_trait;
use futures::{Stream, StreamExt};
use log::info;
use reqwest::{Client as HttpClient, header::{HeaderMap, HeaderValue, CONTENT_TYPE, AUTHORIZATION}};
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{ChatClient, CompletionResponse};
use crate::llm::LlmConfig;
use rllm::builder::LLMBackend;

pub struct GroqChatClient {
    http: HttpClient,
    api_key: String,
    model: String,
    base_url: String,
}

#[derive(Serialize, Deserialize)]
struct GroqMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct GroqRequest {
    messages: Vec<GroqMessage>,
    model: String,
    temperature: f32,
    #[serde(rename = "max_tokens")]
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
}

#[derive(Deserialize)]
struct GroqChoice {
    message: GroqMessage,
}

#[derive(Deserialize)]
struct GroqStreamResponse {
    choices: Vec<GroqStreamChoice>,
}

#[derive(Deserialize)]
struct GroqStreamChoice {
    delta: GroqDelta,
    #[serde(rename = "finish_reason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GroqDelta {
    content: Option<String>,
}

impl GroqChatClient {
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let chat_model = model.unwrap_or_else(|| "llama-3.1-8b-instruct".to_string());
        let api_url = base_url.unwrap_or_else(|| "https://api.groq.com".to_string());
        
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION, 
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| format!("Invalid API key format: {}", e))?
        );
        
        let http = HttpClient::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| Box::new(e) as Box<dyn StdError + Send + Sync>)?;

        Ok(Self {
            http,
            api_key,
            model: chat_model,
            base_url: api_url,
        })
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let api_key = config.api_key
            .clone()
            .ok_or_else(|| "Groq API key is required".to_string())?;
        
        Self::new(
            api_key,
            config.completion_model.clone(),
            config.base_url.clone(),
        )
    }
}

#[async_trait]
impl ChatClient for GroqChatClient {
    async fn complete(
        &self,
        prompt: &str
    ) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>> {
        let url = format!("{}", self.base_url.trim_end_matches('/'));
        
        let messages = vec![GroqMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        
        let req = GroqRequest {
            messages,
            model: self.model.clone(),
            temperature: 0.7,
            max_tokens: 1024,
            stream: None,
        };
        
        let resp = self.http.post(&url)
            .json(&req)
            .send()
            .await?
            .error_for_status()?
            .json::<GroqResponse>()
            .await?;
        
        let content = resp.choices.first()
            .ok_or_else(|| "No response from Groq API".to_string())?
            .message.content.clone();
        
        Ok(CompletionResponse { response: content })
    }
    
    async fn stream_completion(
        &self,
        prompt: &str
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>> {
        let url = format!("{}", self.base_url.trim_end_matches('/'));
        
        let messages = vec![GroqMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        
        let req = GroqRequest {
            messages,
            model: self.model.clone(),
            temperature: 0.7,
            max_tokens: 1024,
            stream: Some(true),
        };
        
        let (tx, rx) = mpsc::channel(32);
        let client = self.http.clone();
        
        info!("Starting Groq stream request to {}", url);
        
        tokio::spawn(async move {
            match client.post(&url).json(&req).send().await {
                Ok(resp) => {
                    if let Err(e) = resp.error_for_status_ref() {
                        let err_msg = format!("Groq API error: {}", e);
                  
                        let _ = tx.send(Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, err_msg)) as _)).await;
                        return;
                    }
                    
                    let mut stream = resp.bytes_stream();
                    
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                if let Ok(text) = String::from_utf8(chunk.to_vec()) {
                                    info!("Groq raw chunk: {}", text);
                                    
                                    for line in text.lines() {
                                        if line.is_empty() || line == "data: [DONE]" {
                                            continue;
                                        }
                                        
                                        if let Some(data) = line.strip_prefix("data: ") {
                                            match serde_json::from_str::<GroqStreamResponse>(data) {
                                                Ok(stream_resp) => {
                                                    for choice in stream_resp.choices {
                                                        if let Some(content) = choice.delta.content {
                                                            if !content.is_empty() {
                                                                if tx.send(Ok(content)).await.is_err() {
                                                                    return;
                                                                }
                                                            }
                                                        }
                                                        
                                                        if let Some(reason) = choice.finish_reason {
                                                            if reason == "stop" {
                                                                return;
                                                            }
                                                        }
                                                    }
                                                },
                                                Err(e) => {
                                                 info!("Failed to parse Groq chunk: {}, error: {}", data, e);
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
                    let err_msg = format!("Groq request error: {}", e);
                    info!("{}", err_msg);
                    let _ = tx.send(Err(Box::new(e) as _)).await;
                }
            }
        });
        
        Ok(Box::pin(ReceiverStream::new(rx)))
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
        Some(self.base_url.clone())
    }
    
    fn get_llm_backend(&self) -> LLMBackend {
        LLMBackend::Groq
    }
}
