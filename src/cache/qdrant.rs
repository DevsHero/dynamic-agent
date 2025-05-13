use crate::cli::Args;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Distance, CreateCollectionBuilder, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParams, value::Kind, vectors_config::Config as VectorsConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CachePayload {
    pub normalized_prompt: String,
    pub response: String,
}

pub async fn init(args: &Args) -> Option<Arc<Qdrant>> {
    if !args.enable_cache {
        return None;
    }
    let client = Qdrant::from_url(&args.cache_qdrant_url)
        .api_key(args.cache_qdrant_api_key.clone())
        .build().ok()?;
    let arc = Arc::new(client);

    let name = &args.cache_qdrant_collection;
    if arc.collection_info(name).await.is_err() {
        let cfg = CreateCollectionBuilder::new(name.clone())
            .vectors_config(VectorsConfig::Params(VectorParams {
                size: args.dimension as u64,
                distance: Distance::Cosine.into(),
                ..Default::default()
            }))
            .build();
        let _ = arc.create_collection(cfg).await;
    }
    Some(arc)
}

pub async fn search(
    client: &Option<Arc<Qdrant>>,
    collection: &str,
    embedding: Vec<f32>,
    threshold: f32,
) -> Option<(String, Vec<f32>)> {
    let cli = client.as_ref()?;
    let resp = cli.search_points(
            SearchPointsBuilder::new(collection, embedding.clone(), 1)
                .with_payload(true)
                .build()
        ).await.ok()?;
    let pt = resp.result.first()?;
    if pt.score < threshold {
        return None;
    }
    if let Some(val) = pt.payload.get("response") {
        if let Some(Kind::StringValue(s)) = &val.kind {
            return Some((s.clone(), embedding));
        }
    }
    let mut smap = serde_json::Map::new();
    for (k, v) in &pt.payload {
        if let Ok(jv) = serde_json::to_value(v) {
            smap.insert(k.clone(), jv);
        }
    }
    if let Ok(cp) = serde_json::from_value::<CachePayload>(JsonValue::Object(smap)) {
        return Some((cp.response, embedding));
    }
    None
}

pub async fn check_qdrant_response(
    client: &Option<Arc<Qdrant>>,
    collection: &str,
    embedding: Vec<f32>,
    threshold: f32,
) -> Result<Option<(String, Vec<f32>)>, Box<dyn std::error::Error>> {
    if let Some((response, embedding)) = search(client, collection, embedding, threshold).await {
        if response.starts_with('{') && response.contains("\"response\"") {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
                if let Some(actual_response) = parsed.get("response").and_then(|v| v.as_str()) {
                    return Ok(Some((actual_response.to_string(), embedding)));
                }
            }
        }
        
        return Ok(Some((response, embedding)));
    }
    Ok(None)
}

pub async fn upsert(
    client: &Option<Arc<Qdrant>>,
    collection: &str,
    normalized: &str,
    response: &str,
    embedding: Vec<f32>,
) {
    let cli = match client.as_ref() {
        Some(c) if !embedding.is_empty() => c,
        _ => return,
    };
    let mut payload = HashMap::new();
    payload.insert(
        "normalized_prompt".to_string(),
        qdrant_client::qdrant::Value {
            kind: Some(Kind::StringValue(normalized.to_string())),
        },
    );
    payload.insert(
        "response".to_string(),
        qdrant_client::qdrant::Value {
            kind: Some(Kind::StringValue(response.to_string())),
        },
    );
    let pt = PointStruct::new(Uuid::new_v4().to_string(), embedding, payload);
    let op = UpsertPointsBuilder::new(collection, vec![pt]).build();
    let _ = cli.upsert_points(op).await;
}