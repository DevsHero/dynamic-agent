use crate::cli::Args;
use redis::{Client, AsyncCommands};
use redis::aio::MultiplexedConnection;
use std::sync::Arc;
use tokio::sync::Mutex;

pub async fn init(args: &Args) -> Option<Arc<Mutex<MultiplexedConnection>>> {
    if !args.enable_cache {
        return None;
    }
    let client = Client::open(args.cache_redis_url.as_str()).ok()?;
    let conn = client.get_multiplexed_async_connection().await.ok()?;
    Some(Arc::new(Mutex::new(conn)))
}

pub async fn get(
    conn: &Option<Arc<Mutex<MultiplexedConnection>>>,
    key: &str
) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(c) = conn {
        let mut guard = c.lock().await;
        match guard.get::<_, String>(key).await {
            Ok(val) => Ok(Some(val)),
            Err(_) => Ok(None),
        }
    } else {
        Ok(None)
    }
}

pub async fn set(
    conn: &Option<Arc<Mutex<MultiplexedConnection>>>,
    key: &str,
    val: &str,
    ttl: usize
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(c) = conn {
        let mut guard = c.lock().await;
        if ttl > 0 {
            guard.set_ex::<_,_,()>(key, val, ttl as u64).await?;
        } else {
            guard.set::<_,_,()>(key, val).await?;
        }
    }
    Ok(())
}
