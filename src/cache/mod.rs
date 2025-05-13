pub mod redis;
pub mod qdrant;

use crate::cli::Args;
use crate::llm::embedding::EmbeddingClient;
use qdrant_client::Qdrant;
use ::redis::aio::MultiplexedConnection;
 
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct CacheClients {
    pub redis: Option<Arc<Mutex<MultiplexedConnection>>>,
    pub qdrant: Option<Arc<Qdrant>>,
    pub collection: String,
    pub threshold: f32,
    pub ttl: usize,
}

pub async fn init(args: &Args) -> CacheClients {
    CacheClients {
        redis: redis::init(args).await,
        qdrant: qdrant::init(args).await,
        collection: args.cache_qdrant_collection.clone(),
        threshold: args.cache_similarity_threshold,
        ttl: args.cache_redis_ttl,
    }
}

pub async fn check(
    clients: &CacheClients,
    normalized: &str,
    embedding_client: &dyn EmbeddingClient,
) -> Result<Option<(String, Vec<f32>)>, Box<dyn std::error::Error + Send + Sync>> {
    // Try Redis first
    if let Some(val) = redis::get(&clients.redis, normalized).await? {
        // Check if Redis value is JSON with response field
        if val.starts_with('{') && val.contains("\"response\"") {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&val) {
                if let Some(response) = json_val.get("response").and_then(|v| v.as_str()) {
                    return Ok(Some((response.to_string(), Vec::new())));
                }
            }
        }
        return Ok(Some((val, Vec::new())));
    }
    
    let emb = embedding_client.embed(normalized).await?.embedding;
    if let Some(hit) = qdrant::search(&clients.qdrant, &clients.collection, emb.clone(), clients.threshold).await {
        let (response_text, emb_vec) = hit;
        
        if response_text.starts_with('{') && response_text.contains("\"response\"") {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&response_text) {
                if let Some(response) = json_val.get("response").and_then(|v| v.as_str()) {
                    return Ok(Some((response.to_string(), emb_vec)));
                }
            }
        }
        return Ok(Some((response_text, emb_vec)));
    }
    
    Ok(None)
}

pub async fn update(
    clients: &CacheClients,
    normalized: &str,
    response: &str,
    embedding: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    redis::set(&clients.redis, normalized, response, clients.ttl).await?;
    qdrant::upsert(&clients.qdrant, &clients.collection, normalized, response, embedding).await;
    Ok(())
}

pub async fn update_streaming(
    clients: &CacheClients,
    query: &str,
    full_response: &str,
    thinking: Option<&str>,
    embedding: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

    redis::set(&clients.redis, query, full_response, clients.ttl).await?;
    
    if let Some(ref _qdrant) = clients.qdrant {
        if thinking.is_some() {
            let combined_response = serde_json::json!({
                "response": full_response,
                "thinking": thinking.unwrap_or(""),
                "is_streaming": true
            }).to_string();
            
            qdrant::upsert(
                &clients.qdrant, 
                &clients.collection, 
                query, 
                &combined_response, 
                embedding
            ).await;
        } else {

            qdrant::upsert(
                &clients.qdrant,
                &clients.collection,
                query,
                full_response,
                embedding
            ).await;
        }
    }
    
    Ok(())
}