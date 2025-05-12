use async_trait::async_trait;
use std::{error::Error as StdError, pin::Pin };
use futures::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use serde::{Deserialize, Serialize};
use log::info;

use super::{ChatClient, CompletionResponse, http_stream_generate};
use crate::llm::LlmConfig; 
use rllm::chat::{ChatMessage, ChatRole, MessageType};
use rllm::builder::{LLMBackend, LLMBuilder};
use rllm::LLMProvider;

 
#[derive(Serialize)]
struct GeminiStreamRequest {
    contents: Vec<GeminiContent>,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize)]
struct GoogleChunk {
    candidates: Vec<GoogleCandidate>,
}

#[derive(Deserialize)]
struct GoogleCandidate {
    content: GoogleContent,
}

#[derive(Deserialize)]
struct GoogleContent {
    parts: Vec<GooglePart>,
}

#[derive(Deserialize)]
struct GooglePart {
    text: String,
}

fn parse_gemini_line(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() || line == "[" || line == "]" || line == "," {
        return None;
    }
    
    if line.starts_with('{') {
       
        let json_obj = if line.ends_with('}') {
            line.to_string()
        } else if line.ends_with("},") {
            line[..line.len()-1].to_string()
        } else {
            return None; 
        };
        
        return serde_json::from_str::<GoogleChunk>(&json_obj)
            .ok()
            .and_then(|gc| {
                gc.candidates.first().and_then(|c| {
                    c.content.parts.first().map(|p| p.text.clone())
                })
            });
    }
    
    if line.contains("\"text\":") {
        let text_part = line.trim();
        if let Some(start) = text_part.find(':') {
            let value_part = &text_part[start+1..].trim();
            if value_part.starts_with('"') && value_part.contains('"') {
                let first_quote = value_part.find('"').unwrap();
                let last_quote = value_part.rfind('"').unwrap();
                if last_quote > first_quote {
                    return Some(value_part[first_quote+1..last_quote].to_string());
                }
            }
        }
    }
    
    None
}

pub struct GeminiChatClient {
    llm: Box<dyn LLMProvider + Send + Sync>,
    api_key: String,
    model: String,
    base_url: Option<String>,
}

impl GeminiChatClient {
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let chat_model = model.unwrap_or_else(|| "gemini-1.5-flash-latest".to_string());

        let mut builder = LLMBuilder::new()
            .backend(LLMBackend::Google)
            .api_key(api_key.clone())
            .model(&chat_model)
            .stream(false);

        if let Some(url) = &base_url {
            builder = builder.base_url(url);
        }
        if let Some(tokens) = max_tokens {
            builder = builder.max_tokens(tokens);
        }
        if let Some(temp) = temperature {
            builder = builder.temperature(temp);
        }

        let llm_provider = builder.build()?;

        Ok(Self { 
            llm: llm_provider,
            api_key,
            model: chat_model,
            base_url
        })
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let api_key = config.api_key
            .clone()
            .ok_or_else(|| "Google API key is required for GeminiChatClient".to_string())?;
        let model = config.completion_model.clone();
        let base_url = config.base_url.clone();
        let max_tokens = None;
        let temperature = None;

        Self::new(api_key, model, base_url, max_tokens, temperature)
    }
}

#[async_trait]
impl ChatClient for GeminiChatClient {
    async fn complete(
        &self,
        prompt: &str
    ) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>> {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: prompt.to_string(),
            message_type: MessageType::Text,
        }];
        info!(
            "GeminiChatClient::complete() → model={} base_url={:?}",
            self.model,
            self.base_url
        );
        let resp = self.llm.chat(&messages).await?;
        let text = resp
            .text()
            .map(|s| s.to_string())
            .unwrap_or_else(|| resp.to_string());
        Ok(CompletionResponse { response: text })
    }

    async fn complete_stream(
        &self,
        prompt: &str
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>,
        Box<dyn StdError + Send + Sync>
    > {
        info!(
            "GeminiChatClient::complete_stream() → model={} configured_base_url={:?}",
            self.model,
            self.base_url 
        );

        let content = GeminiContent {
            parts: vec![GeminiPart {
                text: prompt.to_string()
            }],
        };
        
        let payload = GeminiStreamRequest {
            contents: vec![content],
        };

        let model_specific_base_url = self.base_url.clone().ok_or_else(|| {
            Box::<dyn StdError + Send + Sync>::from(
                "Gemini base_url (CHAT_BASE_URL) is not configured or is empty. It should point to the specific model endpoint.",
            )
        })?;

        let route_suffix = format!(":streamGenerateContent?key={}", self.api_key);
        info!("Attempting to stream from URL: {}{}", model_specific_base_url, route_suffix);

        let headers = vec![
            ("Content-Type".to_string(), "application/json".to_string())
        ];
        
        match http_stream_generate(
            model_specific_base_url,
            &route_suffix,
            payload,
            parse_gemini_line,
            Some(headers),  
        )
        .await
        {
            Ok(stream) => Ok(stream),
            Err(e) => { 
                let resp = self.complete(prompt).await?;
                let text = resp.response;
                let (tx, rx) = mpsc::channel(1);
                tokio::spawn(async move {
                    let _ = tx.send(Ok(text)).await;
                });
                Ok(Box::pin(ReceiverStream::new(rx)))
            }
        }
    }

    async fn stream_completion(
        &self,
        prompt: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>,
        Box<dyn StdError + Send + Sync>
    > {
        info!(
            "GeminiChatClient::stream_completion() → forwarding to complete_stream()"
        );
        self.complete_stream(prompt).await
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
        LLMBackend::Google
    }
}
