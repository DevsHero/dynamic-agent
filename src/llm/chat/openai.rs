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

pub struct OpenAIChatClient {
    http: HttpClient,
    api_key: String,
    model: String,
    base_url: String,
    use_responses_endpoint: bool,
}

#[derive(Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>, 
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Serialize)]
struct OpenAIResponsesRequest {
    model: String,
    input: Vec<String>,
    text: OpenAITextFormat,
    reasoning: serde_json::Value,
    tools: Vec<serde_json::Value>,
    temperature: f32,
    max_output_tokens: u32,
    top_p: f32,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct OpenAITextFormat {
    format: OpenAIFormat,
}

#[derive(Serialize)]
struct OpenAIFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIDelta,
    #[serde(rename = "finish_reason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIResponsesResponse {
    _result: String,
}

#[derive(Deserialize)]
struct OpenAIResponsesStreamResponse {
    delta: Option<String>,
    done: Option<bool>,
}

impl OpenAIChatClient {
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
        use_responses_endpoint: bool,
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let chat_model = model.unwrap_or_else(|| "gpt-4o".to_string());
        let api_url = base_url.unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string());
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
            use_responses_endpoint,
        })
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let api_key = config.api_key
            .clone()
            .ok_or_else(|| "OpenAI API key is required".to_string())?;
        
        let use_responses_endpoint = config.base_url
            .as_ref()
            .map(|url| url.contains("/responses"))
            .unwrap_or(false);
        
        Self::new(
            api_key,
            config.completion_model.clone(),
            config.base_url.clone(),
            use_responses_endpoint,
        )
    }
    
    async fn generate_stream(
        &self,
        prompt: &str
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>> {
        if self.use_responses_endpoint {
            return self.generate_stream_responses(prompt).await;
        } else {
            return self.generate_stream_chat(prompt).await;
        }
    }
    
    async fn generate_stream_chat(
        &self,
        prompt: &str
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>> {
        let url = self.base_url.trim_end_matches('/').to_string();
        
        let messages = vec![OpenAIMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        
        let req = OpenAIChatRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.7,
            max_tokens: Some(2048),
            response_format: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: Some(true),
            store: None,
        };
        
        let (tx, rx) = mpsc::channel(32);
        let client = self.http.clone();
        let auth_header = format!("Bearer {}", self.api_key);
        
        tokio::spawn(async move {
            let resp = match client.post(&url)
                .header(AUTHORIZATION, auth_header)
                .json(&req)
                .send()
                .await {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(Err(Box::new(e) as _)).await;
                        return;
                    }
                };
                
            if let Err(e) = resp.error_for_status_ref() {
                let _ = tx.send(Err(Box::new(e) as _)).await;
                return;
            }
            
            let mut stream = resp.bytes_stream();
            
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Ok(text) = String::from_utf8(chunk.to_vec()) {
                            info!("OpenAI raw chunk: {}", text);
                            
                            for line in text.lines() {
                                if line.is_empty() || line == "data: [DONE]" {
                                    continue;
                                }
                                
                                if let Some(data) = line.strip_prefix("data: ") {
                                    match serde_json::from_str::<OpenAIStreamResponse>(data) {
                                        Ok(stream_resp) => {
                                            for choice in stream_resp.choices {
                                                if let Some(content) = choice.delta.content {
                                                    if !content.is_empty() {
                                                        if tx.send(Ok(content)).await.is_err() {
                                                            return;
                                                        }
                                                    }
                                                }
                                                
                                                if let Some(reason) = &choice.finish_reason {
                                                    if reason == "stop" {
                                                        return;
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
        });
        
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
    
    async fn generate_stream_responses(
        &self,
        prompt: &str
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>> {
        let url = if self.base_url.ends_with("/v1/responses") {
            self.base_url.clone()
        } else {
            format!("{}/v1/responses", self.base_url.trim_end_matches('/'))
        };
        
        let req = OpenAIResponsesRequest {
            model: self.model.clone(),
            input: vec![prompt.to_string()],
            text: OpenAITextFormat {
                format: OpenAIFormat {
                    format_type: "text".to_string(),
                },
            },
            reasoning: serde_json::json!({}),
            tools: Vec::new(),
            temperature: 1.0,
            max_output_tokens: 2048,
            top_p: 1.0,
            store: true,
            stream: Some(true),
        };
        
        let (tx, rx) = mpsc::channel(32);
        let client = self.http.clone();
        let auth_header = format!("Bearer {}", self.api_key);
        
        tokio::spawn(async move {
            let resp = match client.post(&url)
                .header(AUTHORIZATION, auth_header)
                .json(&req)
                .send()
                .await {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(Err(Box::new(e) as _)).await;
                        return;
                    }
                };
                
            if let Err(e) = resp.error_for_status_ref() {
                let _ = tx.send(Err(Box::new(e) as _)).await;
                return;
            }
            
            let mut stream = resp.bytes_stream();
            
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Ok(text) = String::from_utf8(chunk.to_vec()) {
                            info!("OpenAI responses raw chunk: {}", text);
                            
                            for line in text.lines() {
                                if line.is_empty() || line == "data: [DONE]" {
                                    continue;
                                }
                                
                                if let Some(data) = line.strip_prefix("data: ") {
                                    match serde_json::from_str::<OpenAIResponsesStreamResponse>(data) {
                                        Ok(stream_resp) => {
                                            if let Some(delta) = stream_resp.delta {
                                                if !delta.is_empty() {
                                                    if tx.send(Ok(delta)).await.is_err() {
                                                        return;
                                                    }
                                                }
                                            }
                                            
                                            if let Some(done) = stream_resp.done {
                                                if done {
                                                    return;
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
        });
        
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

#[async_trait]
impl ChatClient for OpenAIChatClient {
    async fn complete(
        &self,
        prompt: &str
    ) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>> {
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        
        let messages = vec![OpenAIMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        
        let req = OpenAIChatRequest {
            model: self.model.clone(),
            messages,
            temperature: 1.0,
            response_format: Some(ResponseFormat { format_type: "text".to_string() }),
            max_completion_tokens: Some(2048),
            max_tokens: None,
            top_p: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            stream: None,
            store: Some(false),
        };
        
        let resp = self.http.post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await?
            .error_for_status()?
            .json::<OpenAIResponse>()
            .await?;
        
        let content = resp.choices.first()
            .ok_or_else(|| "No response from OpenAI API".to_string())?
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
        Some(self.base_url.clone())
    }
    
    fn get_llm_backend(&self) -> LLMBackend {
        LLMBackend::OpenAI
    }
}
