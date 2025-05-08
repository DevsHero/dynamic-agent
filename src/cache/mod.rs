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
    if let Some(val) = redis::get(&clients.redis, normalized).await? {
        return Ok(Some((val, Vec::new())));
    }
    let emb = embedding_client.embed(normalized).await?.embedding;
    if let Some(hit) = qdrant::search(&clients.qdrant, &clients.collection, emb.clone(), clients.threshold).await {
        return Ok(Some((hit.0, hit.1)));
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