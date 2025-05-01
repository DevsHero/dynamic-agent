use async_trait::async_trait;
use crate::models::chat::{ ChatMessage, Conversation };
use crate::history::HistoryStore;
use crate::cli::Args;
use std::error::Error;
use chrono::Utc;
use log::error;
use redis::{ Client, AsyncCommands };
use serde::{ Serialize, Deserialize };

#[derive(Serialize, Deserialize)]
struct StoredMessage {
    role: String,
    content: String,
    timestamp: i64,
}

pub struct RedisHistoryStore {
    client: Client,
    key_prefix: String,
    _scan_count: usize,
}

impl RedisHistoryStore {
    pub fn new(args: Args) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Ok(Self {
            client: Client::open(args.history_host.as_str())?,
            key_prefix: args.history_redis_prefix,
            _scan_count: args.history_redis_scan_count,
        })
    }

    async fn get_connection(&self) -> Result<redis::aio::MultiplexedConnection, redis::RedisError> {
        self.client.get_multiplexed_async_connection().await
    }
}

#[async_trait]
impl HistoryStore for RedisHistoryStore {
    async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut conn = self.get_connection().await?;
        let key = format!("{}{}", self.key_prefix, conversation_id);

        let message = StoredMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Utc::now().timestamp(),
        };

        let json_msg = serde_json::to_string(&message)?;
        let _: i64 = conn.lpush(&key, &json_msg).await?;
        Ok(())
    }

    async fn get_conversation(
        &self,
        conversation_id: &str,
        limit: usize
    ) -> Result<Conversation, Box<dyn Error + Send + Sync>> {
        let mut conn = self.get_connection().await?;
        let key = format!("{}{}", self.key_prefix, conversation_id);
        let json_entries: Vec<String> = conn.lrange(&key, 0, (limit as isize) - 1).await?;
        let mut messages = Vec::new();

        for json_entry in &json_entries {
            match serde_json::from_str::<StoredMessage>(json_entry) {
                Ok(msg) => {
                    messages.push(ChatMessage {
                        role: msg.role,
                        content: msg.content,
                        timestamp: msg.timestamp,
                    });
                }
                Err(e) => {
                    error!("Error parsing history entry: {}", e);
                }
            }
        }
        messages.reverse();

        Ok(Conversation {
            id: conversation_id.to_string(),
            messages,
        })
    }
}
