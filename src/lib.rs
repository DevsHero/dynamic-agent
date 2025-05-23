pub mod agent;
pub mod models;
pub mod server; 
pub mod config;
pub mod llm;
pub mod cli;
pub mod history;
pub mod rag;
pub mod cache;

use agent::AIAgent;
use cli::Args;
use config::prompt::initialize_prompt_configuration;
use log::info;
use server::Server;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Mutex;



pub async fn run(args: Args) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("--- Core Configuration ---");
    info!("Server Address: {}", args.server_addr);
    info!("Vector Store Type: {}", args.vector_type);
    info!("Vector Store Host: {}", args.host);
    info!("Chat LLM Type: {}", args.chat_llm_type);
    info!("Embedding LLM Type: {}", args.embedding_llm_type);
    info!("History Store Type: {}", args.history_type);
    info!("History Store Host: {}", args.history_host);
    info!("Schema Path: {}", args.schema_path);
    info!("Prompts Path: {}", args.prompts_path);
    info!("Auto Schema Reload: {}", args.auto_schema);
    info!("LLM Field Resolution: {}", args.llm_query);
    info!("Cache Enabled: {}", args.enable_cache);
    if args.enable_cache {
        info!("Cache Redis URL: {}", args.cache_redis_url);
        info!("Cache Qdrant URL: {}", args.cache_qdrant_url);
        info!("Cache Qdrant Collection: {}", args.cache_qdrant_collection);
    }
    
    if args.enable_remote_prompts {
        info!("Remote Prompts: Enabled");
        info!("Remote Prompts Project ID: {}", args.remote_prompts_project_id.as_deref().unwrap_or("Not specified"));
        info!("Remote Prompts SA Key Path: {}", args.remote_prompts_sa_key_path.as_deref().unwrap_or("Not specified"));
    } else {
        info!("Remote Prompts: Disabled");
    }
    
    info!("-------------------------");
    
    let shared_prompt_config = match initialize_prompt_configuration(&args).await {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to initialize prompt configuration: {e}");
            return Err(Box::new(e));
        }
    };

    let agent_args = args.clone();
    let agent = Arc::new(Mutex::new(AIAgent::new(agent_args, Arc::clone(&shared_prompt_config)).await?));
    
    let addr = args.server_addr.clone();
    info!("Starting WebSocket server on: {addr}" );
    let server = Server::new(
        addr, 
        agent,  
        args.clone()
    );
    server.run().await?;

    Ok(())
}

