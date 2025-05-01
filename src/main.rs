mod agent;
mod models;
mod server;
mod websocket;
mod config;
mod llm;
mod cli;
mod history;
mod rag;
use agent::AIAgent;
use clap::Parser;
use cli::Args;
use dotenv::dotenv;
use server::Server;
use std::error::Error;
use std::sync::Arc;
use log::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    dotenv().ok();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

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
    info!("-------------------------");

    let agent_args = args.clone();
    let agent = Arc::new(AIAgent::new(agent_args).await?);
    let addr = args.server_addr.clone();
    info!("Starting server on: {}", addr);
    let server = Server::new(addr, agent, args.server_api_key.clone(), args.clone());
    server.run().await?;

    Ok(())
}
