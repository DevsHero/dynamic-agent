pub mod chat;
pub mod embedding;
use serde::{ Deserialize, Serialize };
use std::str::FromStr;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmType {
    Ollama,
    OpenAI,
    Anthropic,
    Gemini,
    DeepSeek,
    XAI,
    Groq,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseLlmTypeError {
    message: String,
}

impl fmt::Display for ParseLlmTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseLlmTypeError {}
impl FromStr for LlmType {
    type Err = ParseLlmTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ollama" => Ok(LlmType::Ollama),
            "openai" => Ok(LlmType::OpenAI),
            "anthropic" => Ok(LlmType::Anthropic),
            "gemini" => Ok(LlmType::Gemini),
            "deepseek" => Ok(LlmType::DeepSeek),
            "xai" => Ok(LlmType::XAI),
            "groq" => Ok(LlmType::Groq),
            _ =>
                Err(ParseLlmTypeError {
                    message: format!("Invalid LLM type: '{}'", s),
                }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub llm_type: LlmType,
    pub api_key: Option<String>,
    pub completion_model: Option<String>,
    pub embedding_model: Option<String>,
    pub base_url: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            llm_type: LlmType::Ollama,
            api_key: None,
            completion_model: None,
            embedding_model: None,
            base_url: None,
        }
    }
}

pub fn parse_llm_type(type_str: &str) -> Result<LlmType, String> {
    match type_str.to_lowercase().as_str() {
        "ollama" => Ok(LlmType::Ollama),
        "openai" => Ok(LlmType::OpenAI),
        "anthropic" => Ok(LlmType::Anthropic),
        "gemini" => Ok(LlmType::Gemini),
        "deepseek" => Ok(LlmType::DeepSeek),
        "xai" => Ok(LlmType::XAI),
        "groq" => Ok(LlmType::Groq),
        _ => Err(format!("Unsupported LLM type: {}", type_str)),
    }
}
