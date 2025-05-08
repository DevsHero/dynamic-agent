use yup_oauth2::{ServiceAccountAuthenticator, read_service_account_key};
use std::path::Path;
use reqwest::header::{AUTHORIZATION, ACCEPT, IF_NONE_MATCH};
use std::sync::Mutex;
use crate::config::prompt::{PromptConfig, PromptError, load_prompts_from_str}; 
use crate::cli::Args;
use std::fs;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock as TokioRwLock;
use log::{info, error};
use serde_json::Value as JsonValue;

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    SerdeJson(serde_json::Error),
    Prompt(PromptError),
    RemoteConfig(String),
    MissingRemoteConfig(String),
}

impl From<std::io::Error> for ConfigError {
    fn from(err: std::io::Error) -> Self { ConfigError::Io(err) }
}

impl From<serde_json::Error> for ConfigError {
    fn from(err: serde_json::Error) -> Self { ConfigError::SerdeJson(err) }
}

impl From<PromptError> for ConfigError {
    fn from(err: PromptError) -> Self { ConfigError::Prompt(err) }
}

fn load_prompts_from_file(path: &str) -> Result<Arc<PromptConfig>, ConfigError> {
    let json_str = fs::read_to_string(path)?;
    let mut config: PromptConfig = serde_json::from_str(&json_str)?;
    config.last_loaded = Some(SystemTime::now());
   
    Ok(Arc::new(config))
}

async fn get_access_token(sa_key_path: &str) -> Result<String, PromptError> {
    let key = read_service_account_key(Path::new(sa_key_path))
        .await
        .map_err(|e| PromptError::ActionError(format!("Failed to load SA key from {}: {}", sa_key_path, e)))?;
    
    let auth = ServiceAccountAuthenticator::builder(key)
        .build()
        .await
        .map_err(|e| PromptError::ActionError(e.to_string()))?;
    
    let token = auth
        .token(&["https://www.googleapis.com/auth/firebase.remoteconfig"])
        .await
        .map_err(|e| PromptError::ActionError(e.to_string()))?;
    
    token.token()
        .ok_or_else(|| PromptError::ActionError("OAuth token was None".to_string()))
        .map(|t| t.to_string())
}

pub struct RemoteConfigClient {
    client: reqwest::Client,
    etag: Mutex<Option<String>>,
}

impl RemoteConfigClient {
    pub fn new() -> Self {
        RemoteConfigClient {
            client: reqwest::Client::new(),
            etag: Mutex::new(None),
        }
    }

    pub async fn fetch_config(
        &self,
        project_id: &str,
        sa_key_path: &str
    ) -> Result<Option<String>, PromptError> {
        let token = get_access_token(sa_key_path).await?;
        let url = format!(
            "https://firebaseremoteconfig.googleapis.com/v1/projects/{}/remoteConfig",
            project_id
        );
        let mut req = self.client
            .get(&url)
            .header(AUTHORIZATION, format!("Bearer {}", token))
            .header(ACCEPT, "application/json");

        if let Some(etag) = &*self.etag.lock().unwrap() {
            req = req.header(IF_NONE_MATCH, etag.clone());
        }

        let resp = req.send().await
            .map_err(|e| PromptError::ActionError(e.to_string()))?;

        match resp.status() {
            reqwest::StatusCode::NOT_MODIFIED => Ok(None),
            reqwest::StatusCode::OK => {

                if let Some(etag_val) = resp.headers().get(reqwest::header::ETAG) {
                    if let Ok(etag_str) = etag_val.to_str() {
                        *self.etag.lock().unwrap() = Some(etag_str.to_string());
                    }
                }

                let body_text = resp.text().await
                    .map_err(|e| PromptError::ActionError(e.to_string()))?;
                let root: JsonValue = serde_json::from_str(&body_text)
                    .map_err(|e| PromptError::ActionError(format!("Invalid JSON from remoteConfig: {}", e)))?;

                let prompt_value = root
                    .get("parameters")
                    .and_then(|p| p.get("prompts"))
                    .and_then(|p| p.get("defaultValue"))
                    .and_then(|dv| dv.get("value"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        PromptError::ActionError(
                            "Missing parameters.prompts.defaultValue.value in remoteConfig".into()
                        )
                    })?;

                Ok(Some(prompt_value.to_string()))
            }
            s => {
                let err_body = resp.text().await.unwrap_or_default();
                Err(PromptError::ActionError(
                    format!("Unexpected status {}: {}", s, err_body)
                ))
            }
        }
    }
}

pub async fn initialize_app_config(args: &Args) -> Result<Arc<TokioRwLock<Arc<PromptConfig>>>, ConfigError> {
    let initial_prompts: Arc<PromptConfig>;

    if args.enable_remote_prompts {
        info!("Remote prompts enabled. Attempting to fetch initial configuration...");
        let project_id = args.remote_prompts_project_id.as_deref()
            .ok_or_else(|| ConfigError::MissingRemoteConfig("REMOTE_PROMPTS_PROJECT_ID is required when remote prompts are enabled.".to_string()))?;
        let sa_key_path = args.remote_prompts_sa_key_path.as_deref()
            .ok_or_else(|| ConfigError::MissingRemoteConfig("REMOTE_PROMPTS_SA_KEY_PATH is required when remote prompts are enabled.".to_string()))?;

        let remote_client = RemoteConfigClient::new();

        match remote_client.fetch_config(project_id, sa_key_path).await {
            Ok(Some(json_str)) => {
                info!("Successfully fetched remote prompts.");
                initial_prompts = load_prompts_from_str(&json_str).map_err(ConfigError::Prompt)?;
            }
            Ok(None) => {
                info!("Remote prompts not modified or empty. Falling back to local prompts from: {}", args.prompts_path);
                initial_prompts = load_prompts_from_file(&args.prompts_path)?;
            }
            Err(e) => {
                error!("Failed to fetch remote prompts: {:?}. Falling back to local prompts from: {}", e, args.prompts_path);
                initial_prompts = load_prompts_from_file(&args.prompts_path)?;
            }
        }
    } else {
        info!("Loading local prompts from: {}", args.prompts_path);
        initial_prompts = load_prompts_from_file(&args.prompts_path)?;
    }

    let shared_prompts = Arc::new(TokioRwLock::new(initial_prompts));

    Ok(shared_prompts)
}
