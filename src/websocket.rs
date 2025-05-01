use crate::{ agent::AIAgent, cli::Args, models::websocket::{ ClientMessage, ServerMessage } };
use chrono::Utc;
use clap::Parser;
use futures::{ SinkExt, StreamExt };
use log::{ info, warn, error };
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::io::{ AsyncRead, AsyncWrite };
use tokio_tungstenite::{ tungstenite::protocol::Message, WebSocketStream };
use uuid::Uuid;
const MAX_MESSAGE_SIZE: usize = 1 * 1024 * 1024;

pub async fn handle_connection<S>(
    peer: SocketAddr,
    websocket: WebSocketStream<S>,
    agent: Arc<Mutex<AIAgent>>
)
    where S: AsyncRead + AsyncWrite + Unpin
{
    let args = Args::parse();
    info!("New WebSocket connection: {}", peer);

    let mut agent_guard = agent.lock().await;
    if let Err(e) = agent_guard.reload_prompts_if_changed(&args.clone()).await {
        error!("Failed to reload prompts: {}", e);
    }
    if args.auto_schema {
        info!("Auto-schema generation enabled. Checking if schema needs reloading...");
        if let Err(e) = agent_guard.reload_schema_if_needed(&args.clone()).await {
            error!("Failed to reload schema: {}", e);
        }
    } else {
        info!("Auto-schema generation disabled. Skipping schema reload check.");
    }
    drop(agent_guard);

    let (mut tx, mut rx) = websocket.split();
    let conversation_id = Uuid::new_v4().to_string();
    info!("Assigned conversation ID {} to {}", conversation_id, peer);

    while let Some(msg) = rx.next().await {
        match msg {
            Ok(message) => {
                if message.len() > MAX_MESSAGE_SIZE {
                    warn!(
                        "Message from {} exceeds size limit ({} > {})",
                        peer,
                        message.len(),
                        MAX_MESSAGE_SIZE
                    );
                    let error_msg = ServerMessage::Error {
                        message: "Message too large".to_string(),
                    };
                    let json = serde_json::to_string(&error_msg).unwrap();
                    if tx.send(Message::Text(json)).await.is_err() {
                        error!("Failed to send size limit error to {}", peer);
                    }
                    break;
                }

                match message {
                    Message::Text(text) => {
                        match serde_json::from_str::<ClientMessage>(&text) {
                            Ok(ClientMessage::Chat { content }) => {
                                let processing_msg = ServerMessage::Processing;
                                let processing_json = serde_json
                                    ::to_string(&processing_msg)
                                    .unwrap();
                                if let Err(e) = tx.send(Message::Text(processing_json)).await {
                                    error!("Error sending processing status to {}: {}", peer, e);
                                    break;
                                }

                                let response_result = agent
                                    .lock().await
                                    .process_message(&conversation_id, &content).await;

                                let response_timestamp = Utc::now().timestamp();

                                match response_result {
                                    Ok(response_content) => {
                                        let server_msg = ServerMessage::Response {
                                            content: response_content,
                                            timestamp: response_timestamp,
                                        };
                                        let json = serde_json::to_string(&server_msg).unwrap();
                                        if let Err(e) = tx.send(Message::Text(json)).await {
                                            error!("Error sending message to {}: {}", peer, e);
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        let error_message =
                                            format!("Error processing message: {}", e);
                                        error!(
                                            "Agent processing error for {}: {}",
                                            peer,
                                            error_message
                                        );
                                        let error_msg = ServerMessage::Error {
                                            message: error_message,
                                        };
                                        let json = serde_json::to_string(&error_msg).unwrap();
                                        if let Err(e_inner) = tx.send(Message::Text(json)).await {
                                            error!(
                                                "Error sending error message to {}: {}",
                                                peer,
                                                e_inner
                                            );
                                            break;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Failed to parse message from {}: {}", peer, e);
                                let error_msg = ServerMessage::Error {
                                    message: format!("Failed to parse message: {}", e),
                                };
                                let json = serde_json::to_string(&error_msg).unwrap();
                                if let Err(e) = tx.send(Message::Text(json)).await {
                                    error!("Error sending parse error to {}: {}", peer, e);
                                    break;
                                }
                            }
                        }
                    }
                    Message::Close(_) => {
                        info!("Received close frame from {}", peer);
                        break;
                    }
                    Message::Ping(ping_data) => {
                        if tx.send(Message::Pong(ping_data)).await.is_err() {
                            error!("Failed to send pong to {}", peer);
                            break;
                        }
                    }
                    Message::Pong(_) => {/* Usually ignore pongs */}
                    Message::Binary(_) => {
                        warn!("Ignoring binary message from {}", peer);
                    }
                    Message::Frame(_) => {/* Usually ignore raw frames */}
                }
            }
            Err(e) => {
                match e {
                    | tokio_tungstenite::tungstenite::Error::ConnectionClosed
                    | tokio_tungstenite::tungstenite::Error::Protocol(_)
                    | tokio_tungstenite::tungstenite::Error::Utf8 => {
                        info!("WebSocket connection closed or protocol error for {}: {}", peer, e);
                    }
                    tokio_tungstenite::tungstenite::Error::Io(ref io_err) if
                        io_err.kind() == std::io::ErrorKind::ConnectionReset
                    => {
                        info!("WebSocket connection reset by peer {}", peer);
                    }
                    tokio_tungstenite::tungstenite::Error::Capacity(ref cap_err) => {
                        error!("WebSocket capacity error for {}: {}", peer, cap_err);
                        let error_msg = ServerMessage::Error {
                            message: "Server capacity error".to_string(),
                        };
                        let json = serde_json::to_string(&error_msg).unwrap();
                        let _ = tx.send(Message::Text(json)).await;
                    }
                    _ => {
                        error!("Error receiving message from {}: {}", peer, e);
                    }
                }
                break;
            }
        }
    }
    info!("WebSocket connection closed for {} (Conv ID: {})", peer, conversation_id);
}
