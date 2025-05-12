pub mod ollama;
pub mod openai;
pub mod gemini;
pub mod anthropic;
pub mod deepseek;
pub mod groq;
pub mod xai;

use async_trait::async_trait;
use futures::{Stream, StreamExt, Future}; 
use serde::Deserialize;
use std::error::Error as StdError;
use std::pin::Pin;
use std::sync::Arc;
use super::{ LlmConfig, LlmType };
use self::ollama::OllamaClient;
use self::openai::OpenAIChatClient;
use self::gemini::GeminiChatClient;
use self::anthropic::AnthropicChatClient;
use self::deepseek::DeepSeekChatClient;
use self::groq::GroqChatClient;
use self::xai::XAIChatClient;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use rllm::{
    builder::{ LLMBackend, LLMBuilder },
    chat::{ ChatMessage, ChatRole, MessageType },
};
use reqwest;

#[derive(Deserialize, Debug, Clone)]
pub struct CompletionResponse {
    pub response: String,
}

#[async_trait]
pub trait ChatClient: Send + Sync {
    async fn complete(
        &self,
        prompt: &str
    ) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>>;
    
    async fn complete_stream(
        &self,
        prompt: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>,
        Box<dyn StdError + Send + Sync>
    >
    where
        Self: Sized,
    {
        stream_chat_for_provider(self, prompt).await
    }
    
    async fn stream_completion(
        &self,
        prompt: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>,
        Box<dyn StdError + Send + Sync>
    > {
        stream_chat_for_provider(self, prompt).await
    }
    
    fn get_api_key(&self) -> String;
    fn get_model(&self) -> String;
    fn get_base_url(&self) -> Option<String>;
    fn get_llm_backend(&self) -> LLMBackend;
    fn supports_native_streaming(&self) -> bool {
        false  
    }
}

pub async fn stream_chat_for_provider<T: ChatClient + ?Sized>(
    client: &T,
    prompt: &str
) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>> {
    let api_key = client.get_api_key();
    let model = client.get_model();
    let base_url = client.get_base_url();
    let backend = client.get_llm_backend();
    let supports_streaming = client.supports_native_streaming();

    if supports_streaming {
        return client.stream_completion(prompt).await;
    }

    let api_key = api_key.to_string();
    let model = model.to_string();
    let base_url_clone = base_url.clone();
    let prompt_owned = prompt.to_string();
    
    full_response_as_stream(move || async move {
        let mut builder = LLMBuilder::new()
            .backend(backend)
            .api_key(api_key)
            .model(&model)
            .stream(true);
            
        if let Some(url) = base_url_clone {
            builder = builder.base_url(url);
        }
        
        let provider = builder.build()?;
        
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: prompt_owned,
            message_type: MessageType::Text,
        }];
        
        provider.chat(&messages).await
            .map_err(|e| Box::new(e) as _)
            .map(|resp| {
                resp.text()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| resp.to_string())
            })
    })
}



pub fn create_streaming_response<F, Fut>(
    response_fn: F
) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>>
where
    F: FnOnce(mpsc::Sender<Result<String, Box<dyn StdError + Send + Sync>>>) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = mpsc::channel(32);
    
    tokio::spawn(async move {
        response_fn(tx).await;
    });
    
    Ok(Box::pin(ReceiverStream::new(rx)))
}

pub fn full_response_as_stream<F, Fut>(
    response_fn: F
) -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>, Box<dyn StdError + Send + Sync>>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<String, Box<dyn StdError + Send + Sync>>> + Send + 'static,
{
    create_streaming_response(move |tx| async move {
        match response_fn().await {
            Ok(response) => {
                let _ = tx.send(Ok(response)).await;
            }
            Err(e) => {
                let _ = tx.send(Err(e)).await;
            }
        }
    })
}

pub fn new_client(
    config: &LlmConfig
) -> Result<Arc<dyn ChatClient>, Box<dyn StdError + Send + Sync>> {
    let client: Arc<dyn ChatClient> = match config.llm_type {
        LlmType::Ollama => {
            let specific_client = OllamaClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::OpenAI => {
            let specific_client = OpenAIChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::Gemini => {
            let specific_client = GeminiChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::Anthropic => {
            let specific_client = AnthropicChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::DeepSeek => {
            let specific_client = DeepSeekChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::Groq => {
            let specific_client = GroqChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::XAI => {
            let specific_client = XAIChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
    };
    Ok(client)
}

pub async fn http_stream_generate(
    base_url: String,
    route: &str,           
    payload: impl serde::Serialize + Send + 'static,
    line_parser: fn(&str) -> Option<String>,
    headers: Option<Vec<(String, String)>>,
) -> Result<
    Pin<Box<dyn Stream<Item = Result<String, Box<dyn StdError + Send + Sync>>> + Send>>,
    Box<dyn StdError + Send + Sync>
> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), route);
    let (tx, rx) = mpsc::channel(32);
    let client = reqwest::Client::new();
    
    tokio::spawn(async move {
        let mut req = client.post(&url).json(&payload);
        
        if let Some(header_list) = headers {
            for (name, value) in header_list {
                req = req.header(name, value);
            }
        }
        
        match req.send().await {
            Ok(resp) => {
                if let Err(e) = resp.error_for_status_ref() {
                    let _ = tx.send(Err(Box::new(e) as _)).await;
                    return;
                }
                let mut bytes = resp.bytes_stream();
                while let Some(chunk) = bytes.next().await {
                    match chunk {
                        Ok(buf) => {
                            if let Ok(text) = String::from_utf8(buf.to_vec()) {
                            
                                for line in text.lines() {
                                    if let Some(tok) = line_parser(line) {
                                        if tx.send(Ok(tok)).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(Box::new(e) as _)).await;
                            return;
                        }
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(Box::new(e) as _)).await;
            }
        }
    });
    
    Ok(Box::pin(ReceiverStream::new(rx)))
}
