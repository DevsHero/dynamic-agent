use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock; 
use std::time::SystemTime;
use log::info;
use std::sync::Mutex;
use crate::cli::Args;
use crate::config::remote_config::RemoteConfigClient;

#[derive(Debug)]
pub enum PromptError {
    TemplateNotFound(String),
    IntentNotFound(String),
    ActionError(String),
    IoError(std::io::Error),
    JsonError(serde_json::Error),
    MissingRemoteConfigField(String),
    RemoteFetchError(String),
}

impl fmt::Display for PromptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PromptError::TemplateNotFound(key) => write!(f, "Prompt template '{}' not found", key),
            PromptError::IntentNotFound(key) => write!(f, "Intent definition '{}' not found", key),
            PromptError::ActionError(msg) => write!(f, "Action processing error: {}", msg),
            PromptError::IoError(e) => write!(f, "Prompt file IO error: {}", e),
            PromptError::JsonError(e) => write!(f, "Prompt JSON parsing error: {}", e),
            PromptError::MissingRemoteConfigField(field) => write!(f, "Missing remote configuration field: {}", field),
            PromptError::RemoteFetchError(msg) => write!(f, "Remote prompt fetch error: {}", msg),
        }
    }
}

impl Error for PromptError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PromptError::IoError(e) => Some(e),
            PromptError::JsonError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PromptError {
    fn from(err: std::io::Error) -> Self {
        PromptError::IoError(err)
    }
}

impl From<serde_json::Error> for PromptError {
    fn from(err: serde_json::Error) -> Self {
        PromptError::JsonError(err)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct IntentDefinition {
    pub description: String,
    pub action: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct PromptConfig {
    pub intents: HashMap<String, IntentDefinition>,
    pub query_templates: HashMap<String, String>,
    pub response_templates: HashMap<String, String>,
    #[serde(skip)]
    pub last_loaded: Option<SystemTime>,
}

impl PromptConfig {
    fn _validate(&self) -> Result<(), PromptError> {
        if !self.query_templates.contains_key("intent_classification") {
            return Err(
                PromptError::TemplateNotFound("query_templates:intent_classification".to_string())
            );
        }
        if !self.query_templates.contains_key("rag_topic_inference") {
            return Err(
                PromptError::TemplateNotFound("query_templates:rag_topic_inference".to_string())
            );
        }
        if !self.query_templates.contains_key("rag_dynamic_query_generation") {
            return Err(
                PromptError::TemplateNotFound(
                    "query_templates:rag_dynamic_query_generation".to_string()
                )
            );
        }

        if !self.response_templates.contains_key("rag_final_answer") {
            return Err(
                PromptError::TemplateNotFound("response_templates:rag_final_answer".to_string())
            );
        }
        Ok(())
    }
}

pub fn load_prompts_from_str(json_str: &str) -> Result<Arc<PromptConfig>, PromptError> {
    let mut config: PromptConfig = serde_json::from_str(json_str)?;
    config.last_loaded = Some(SystemTime::now());
    config._validate()?;
    Ok(Arc::new(config))
}

fn load_prompts_from_file_internal(path: &str) -> Result<Arc<PromptConfig>, PromptError> {
    info!("Loading local prompts from: {}", path);
    let file_content = fs::read_to_string(path)?;
    load_prompts_from_str(&file_content)
}

pub fn load_prompts(path: &str) -> Result<Arc<PromptConfig>, Box<dyn Error + Send + Sync>> {
    info!("Using legacy load_prompts function with path: {}", path);
    match load_prompts_from_file_internal(path) {
        Ok(config) => Ok(config),
        Err(e) => Err(Box::new(e)),
    }
}

pub async fn initialize_prompt_configuration(
    args: &Args,
) -> Result<Arc<RwLock<Arc<PromptConfig>>>, PromptError> {
    let initial_prompts: Arc<PromptConfig>;

    if args.enable_remote_prompts {
        info!("Remote prompts enabled. Attempting to fetch initial configuration...");
        let project_id = args.remote_prompts_project_id.as_deref().ok_or_else(|| {
            PromptError::MissingRemoteConfigField("REMOTE_PROMPTS_PROJECT_ID".to_string())
        })?;
        let sa_key_path = args.remote_prompts_sa_key_path.as_deref().ok_or_else(|| {
            PromptError::MissingRemoteConfigField("REMOTE_PROMPTS_SA_KEY_PATH".to_string())
        })?;

        let remote_client = RemoteConfigClient::new();

        match remote_client.fetch_config(project_id, sa_key_path).await {
            Ok(Some(json_str)) => {
                info!("Successfully fetched remote prompts.");
                initial_prompts = load_prompts_from_str(&json_str)?;
            }
            Ok(None) => {
                info!(
                    "Remote prompts not modified or empty. Falling back to local prompts from: {}",
                    args.prompts_path
                );
                initial_prompts = load_prompts_from_file_internal(&args.prompts_path)?;
            }
            Err(e) => {
                eprintln!(
                    "Failed to fetch remote prompts: {:?}. Falling back to local prompts from: {}",
                    e, args.prompts_path
                );
                initial_prompts = load_prompts_from_file_internal(&args.prompts_path)?;
            }
        }
    } else {
        initial_prompts = load_prompts_from_file_internal(&args.prompts_path)?;
    }

    let shared_prompts = Arc::new(RwLock::new(initial_prompts));

    Ok(shared_prompts)
}

pub fn reload_prompts_if_changed<P: AsRef<Path>>(
    path: P,
    current_config: &Arc<PromptConfig>
) -> Result<Option<Arc<PromptConfig>>, PromptError> {
    let metadata = fs::metadata(&path)?;

    if let Ok(modified) = metadata.modified() {
        if let Some(last_loaded) = current_config.last_loaded {
            if modified > last_loaded {
                info!("Prompts file changed, reloading...");
                let new_config = load_prompts_from_file_internal(path.as_ref().to_str().unwrap())?;
                return Ok(Some(new_config));
            }
        } else {
            info!("No last_loaded timestamp, reloading prompts...");
            let new_config = load_prompts_from_file_internal(path.as_ref().to_str().unwrap())?;
            return Ok(Some(new_config));
        }
    }
    Ok(None)
}

fn get_query_template<'a>(config: &'a PromptConfig, key: &str) -> Result<&'a str, PromptError> {
    config.query_templates
        .get(key)
        .map(|s| s.as_str())
        .ok_or_else(|| PromptError::TemplateNotFound(format!("query_templates:{}", key)))
}

fn get_response_template<'a>(config: &'a PromptConfig, key: &str) -> Result<&'a str, PromptError> {
    config.response_templates
        .get(key)
        .map(|s| s.as_str())
        .ok_or_else(|| PromptError::TemplateNotFound(format!("response_templates:{}", key)))
}

pub fn get_intent_prompt(config: &PromptConfig, message: &str) -> Result<String, PromptError> {
    let template = get_query_template(config, "intent_classification")?;

    let descriptions = config.intents
        .iter()
        .map(|(name, definition)| format!("- {}: {}", name, definition.description))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(template.replace("{intent_descriptions}", &descriptions).replace("{message}", message))
}

pub fn get_rag_topic_prompt(
    config: &PromptConfig,
    schema_json: &str,
    user_question: &str
) -> Result<String, PromptError> {
    let template = get_query_template(config, "rag_topic_inference")?;
    Ok(template.replace("{schema_json}", schema_json).replace("{user_question}", user_question))
}

pub fn get_rag_final_prompt(
    config: &PromptConfig,
    schema: &str,
    topic: &str,
    documents: &str,
    user_question: &str
) -> Result<String, PromptError> {
    let template = get_response_template(config, "rag_final_answer")?;

    Ok(
        template
            .replace("{schema}", schema)
            .replace("{topic}", topic)
            .replace("{documents}", documents)
            .replace("{user_question}", user_question)
    )
}

pub fn get_fallback_topic_prompt(
    config: &PromptConfig,
    schema_summary: &str,
    user_question: &str
) -> Result<String, PromptError> {
    let template = get_query_template(config, "fallback_topic_resolver")?;
    
    Ok(template
        .replace("{schema_summary}", schema_summary)
        .replace("{user_question}", user_question))
}


pub fn check_local_prompt_file_changed(path: &str) -> Result<bool, PromptError> {
    let metadata = fs::metadata(path)
        .map_err(|e| PromptError::IoError(e))?;
    
    let modified = metadata.modified()
        .map_err(|e| PromptError::IoError(e))?;
    
    static LAST_MODIFIED: Mutex<Option<SystemTime>> = Mutex::new(None);
    let mut last_mod = LAST_MODIFIED.lock().unwrap();
    
    let changed = match *last_mod {
        Some(time) if modified > time => {
            *last_mod = Some(modified);
            true
        }
        None => {
            *last_mod = Some(modified);
            true
        }
        _ => false
    };
    
    Ok(changed)
}
