use async_trait::async_trait;
use log::info;
use crate::models::chat::{ ChatMessage, Conversation };
use crate::history::HistoryStore;
use crate::cli::Args;
use crate::llm::embedding::EmbeddingClient;
use std::error::Error;
use chrono::Utc;
use std::collections::{ HashMap, HashSet };
use uuid::Uuid;
use qdrant_client::qdrant::Value as QdrantValue;
use std::sync::Arc;

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollection,
    Distance,
    PointStruct,
    ScrollPoints,
    SearchPoints,
    VectorParams,
    VectorsConfig,
    Condition,
    Filter,
    PointId,
    with_payload_selector::SelectorOptions as WithPayloadOptions,
    WithPayloadSelector,
    FieldType,
    CreateFieldIndexCollection,
    OrderBy,
    Direction,
    UpsertPoints,
};

pub struct QdrantHistoryStore {
    client: Qdrant,
    collection_name: String,
    embedding_client: Arc<dyn EmbeddingClient>,
    vector_dim: u64,
}

impl QdrantHistoryStore {
    pub fn new(
        args: Args,
        embedding_client: Arc<dyn EmbeddingClient>
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let client = Qdrant::from_url(&args.history_host).build()?;
        let vector_dim = args.dimension as u64;

        let store = Self {
            client,
            collection_name: args.indexes.clone(),
            embedding_client,
            vector_dim,
        };

        Ok(store)
    }

    async fn ensure_collection_exists(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if !self.client.collection_exists(&self.collection_name).await? {
            self.client.create_collection(CreateCollection {
                collection_name: self.collection_name.clone(),
                vectors_config: Some(
                    VectorsConfig::from(VectorParams {
                        size: self.vector_dim,
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    })
                ),
                ..Default::default()
            }).await?;
            info!("Created Qdrant history collection: {}", self.collection_name);

            self.client.create_field_index(CreateFieldIndexCollection {
                collection_name: self.collection_name.clone(),
                field_name: "timestamp".to_string(),
                field_type: Some(FieldType::Integer.into()),
                wait: Some(true),
                ..Default::default()
            }).await?;
            info!("Created 'timestamp' index in {}", self.collection_name);

            self.client.create_field_index(CreateFieldIndexCollection {
                collection_name: self.collection_name.clone(),
                field_name: "conversation_id".to_string(),
                field_type: Some(FieldType::Keyword.into()),
                wait: Some(true),
                ..Default::default()
            }).await?;
            info!("Created 'conversation_id' index in {}", self.collection_name);
        }
        Ok(())
    }

    fn create_conversation_filter(&self, conversation_id: &str) -> Filter {
        Filter::must([Condition::matches("conversation_id", conversation_id.to_string())])
    }

    fn payload_to_chat_message(payload: HashMap<String, QdrantValue>) -> Option<ChatMessage> {
        let role = payload.get("role")?.as_str()?.to_string();
        let content = payload.get("content")?.as_str()?.to_string();
        let timestamp = payload.get("timestamp")?.as_integer()?;

        Some(ChatMessage { role, content, timestamp })
    }

    fn string_to_point_id(s: &str) -> PointId {
        if let Ok(num) = s.parse::<u64>() {
            return PointId {
                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)),
            };
        }

        PointId {
            point_id_options: Some(
                qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s.to_string())
            ),
        }
    }
}

#[async_trait]
impl HistoryStore for QdrantHistoryStore {
    async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.ensure_collection_exists().await?;

        let timestamp = Utc::now().timestamp();
        let embedding_response = self.embedding_client.embed(content).await?;
        let vector = embedding_response.embedding;

        if (vector.len() as u64) != self.vector_dim {
            return Err(
                format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    self.vector_dim,
                    vector.len()
                ).into()
            );
        }

        let mut payload = HashMap::new();
        payload.insert("conversation_id".to_string(), conversation_id.to_string().into());
        payload.insert("role".to_string(), role.to_string().into());
        payload.insert("content".to_string(), content.to_string().into());
        payload.insert("timestamp".to_string(), timestamp.into());

        let point_id = Uuid::new_v4().to_string();
        let point = PointStruct::new(point_id, vector, payload);
        let upsert_request = UpsertPoints {
            collection_name: self.collection_name.clone(),
            wait: Some(true),
            points: vec![point],
            ordering: None,
            shard_key_selector: None,
        };
        self.client.upsert_points(upsert_request).await?;

        Ok(())
    }

    async fn get_conversation(
        &self,
        conversation_id: &str,
        limit: usize
    ) -> Result<Conversation, Box<dyn Error + Send + Sync>> {
        self.ensure_collection_exists().await?;

        let conversation_filter = self.create_conversation_filter(conversation_id);
        let mut combined_messages: HashMap<String, ChatMessage> = HashMap::new();
        let mut retrieved_ids_str: HashSet<String> = HashSet::new();
        let recency_limit = (limit / 2).max(1);

        let recency_scroll = ScrollPoints {
            collection_name: self.collection_name.clone(),
            filter: Some(conversation_filter.clone()),
            limit: Some(recency_limit as u32),
            with_payload: Some(WithPayloadSelector {
                selector_options: Some(WithPayloadOptions::Enable(true)),
            }),
            order_by: Some(OrderBy {
                key: "timestamp".to_string(),
                direction: Some(Direction::Desc.into()),
                ..Default::default()
            }),
            ..Default::default()
        };

        let recency_response = self.client.scroll(recency_scroll).await?;
        let mut last_message_content: Option<String> = None;

        for point in recency_response.result {
            if let Some(message) = Self::payload_to_chat_message(point.payload) {
                if last_message_content.is_none() {
                    last_message_content = Some(message.content.clone());
                }

                let point_id_str = match &point.id {
                    Some(point_id) =>
                        match &point_id.point_id_options {
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) =>
                                uuid.clone(),
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) =>
                                num.to_string(),
                            _ => String::new(),
                        }
                    None => String::new(),
                };

                if !point_id_str.is_empty() {
                    retrieved_ids_str.insert(point_id_str.clone());
                    combined_messages.insert(point_id_str, message);
                }
            }
        }

        let semantic_limit = limit - combined_messages.len();
        if semantic_limit > 0 && last_message_content.is_some() {
            let query_text = last_message_content.unwrap();
            let query_embedding = self.embedding_client.embed(&query_text).await?.embedding;

            let mut semantic_filter = conversation_filter.clone();
            if !retrieved_ids_str.is_empty() {
                let point_ids_to_exclude: Vec<PointId> = retrieved_ids_str
                    .iter()
                    .map(|s| Self::string_to_point_id(s))
                    .collect();

                if !point_ids_to_exclude.is_empty() {
                    let exclude_filter = Filter {
                        must: vec![],
                        must_not: vec![Condition::has_id(point_ids_to_exclude)],
                        should: vec![],
                        min_should: None,
                    };

                    semantic_filter = Filter {
                        must: vec![
                            Condition {
                                condition_one_of: Some(
                                    qdrant_client::qdrant::condition::ConditionOneOf::Filter(
                                        conversation_filter
                                    )
                                ),
                            },
                            Condition {
                                condition_one_of: Some(
                                    qdrant_client::qdrant::condition::ConditionOneOf::Filter(
                                        exclude_filter
                                    )
                                ),
                            }
                        ],
                        must_not: vec![],
                        should: vec![],
                        min_should: None,
                    };
                }
            }

            let semantic_search = SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: query_embedding,
                filter: Some(semantic_filter),
                limit: semantic_limit as u64,
                with_payload: Some(WithPayloadSelector {
                    selector_options: Some(WithPayloadOptions::Enable(true)),
                }),
                ..Default::default()
            };

            let semantic_response = self.client.search_points(semantic_search).await?;

            for scored_point in semantic_response.result {
                let point_id_str = match &scored_point.id {
                    Some(point_id) =>
                        match &point_id.point_id_options {
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) =>
                                uuid.clone(),
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) =>
                                num.to_string(),
                            _ => String::new(),
                        }
                    None => String::new(),
                };

                if !point_id_str.is_empty() && !combined_messages.contains_key(&point_id_str) {
                    if let Some(message) = Self::payload_to_chat_message(scored_point.payload) {
                        combined_messages.insert(point_id_str, message);
                    }
                }
            }
        }

        let mut final_messages: Vec<ChatMessage> = combined_messages.into_values().collect();
        final_messages.sort_by_key(|m| m.timestamp);

        if final_messages.len() > limit {
            final_messages = final_messages.split_off(final_messages.len() - limit);
        }

        Ok(Conversation {
            id: conversation_id.to_string(),
            messages: final_messages,
        })
    }
}
