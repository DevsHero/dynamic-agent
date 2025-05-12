use crate::agent::AIAgent;
use crate::cli::Args;
use crate::models::websocket::{ClientMessage, ServerMessage};

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::collections::HashMap;

use tokio::sync::Mutex;
use tokio::net::TcpListener;
use tokio::io::{AsyncRead, AsyncWrite};

use tokio_tungstenite::{accept_hdr_async, WebSocketStream};
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response, ErrorResponse};
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio_rustls::TlsAcceptor;

use rustls::ServerConfig;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls_pemfile::{certs, pkcs8_private_keys};

use lazy_static::lazy_static;
use governor::{RateLimiter, Quota, state::{InMemoryState, NotKeyed}, clock::DefaultClock};

use hmac::{Hmac, Mac};
use sha2::Sha256;
use chrono::Utc;
use hex;
use url::form_urlencoded;

use log::{info, warn, error};
use futures::{SinkExt, StreamExt};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

const MAX_MESSAGE_SIZE: usize = 1 * 1024 * 1024;

lazy_static! {
    static ref CONNECTION_LIMITER: RateLimiter<NotKeyed, InMemoryState, DefaultClock> =
        RateLimiter::direct(Quota::per_second(NonZeroU32::new(10).unwrap()));
}

fn load_tls_config(
    cert_path: &str,
    key_path: &str
) -> Result<Arc<ServerConfig>, Box<dyn Error + Send + Sync>> {
    let cert_file = File::open(cert_path).map_err(|e|
        format!("Failed to open TLS certificate file '{}': {}", cert_path, e)
    )?;
    let key_file = File::open(key_path).map_err(|e|
        format!("Failed to open TLS key file '{}': {}", key_path, e)
    )?;

    let mut cert_reader = BufReader::new(cert_file);
    let mut key_reader = BufReader::new(key_file);
    let cert_chain: Vec<CertificateDer<'static>> = certs(&mut cert_reader)
        .collect::<Result<_, _>>()
        .map_err(|e| format!("Failed to read certificate(s): {}", e))?;

    let mut keys = pkcs8_private_keys(&mut key_reader);
    let key = match keys.next() {
        Some(Ok(k)) => PrivateKeyDer::Pkcs8(k),
        Some(Err(e)) => {
            return Err(format!("Error reading private key: {}", e).into());
        }
        None => {
            return Err("No PKCS8 private key found in key file".into());
        }
    };

    let config = ServerConfig::builder().with_no_client_auth().with_single_cert(cert_chain, key)?;
    Ok(Arc::new(config))
}

pub async fn start_ws_server(
    addr: &str,
    agent: Arc<Mutex<AIAgent>>,
    api_key: Option<String>,
    args: Args,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let listener = TcpListener::bind(addr).await?;

    let protocol = if 
        args.enable_tls && 
        args.tls_cert_path.is_some() && 
        args.tls_key_path.is_some() 
    {
        "wss"
    } else {
        "ws"
    };
    info!("{} server listening on: {}", protocol.to_uppercase(), addr);

    let tls_acceptor = if args.enable_tls {
        match (&args.tls_cert_path, &args.tls_key_path) {
            (Some(cert_path), Some(key_path)) => {
                info!(
                    "TLS enabled. Loading certificate from '{}' and key from '{}'",
                    cert_path,
                    key_path
                );
                let config = load_tls_config(cert_path, key_path)?;
                Some(TlsAcceptor::from(config))
            }
            (Some(_), None) | (None, Some(_)) => {
                error!("Both --tls-cert-path and --tls-key-path must be provided to enable TLS.");
                return Err("Missing TLS certificate or key path".into());
            }
            (None, None) => {
                error!("--enable-tls was set but no certificate/key paths provided.");
                return Err("TLS enabled without cert/key".into());
            }
        }
    } else {
        info!("TLS not enabled. Running plain WebSocket (WS) server.");
        None
    };

    loop {
        let (stream, peer) = listener.accept().await?;

        if let Err(_) = CONNECTION_LIMITER.check() {
            warn!("Global connection rate limit exceeded for {}. Dropping connection.", peer);
            continue;
        }

        info!("Incoming connection from: {}", peer);
        let agent_clone = Arc::clone(&agent);
        let required_api_key = api_key.clone();
        let tls_acceptor_clone = tls_acceptor.clone();

        tokio::spawn(async move {
            let process_result = if let Some(acceptor) = tls_acceptor_clone {
                match acceptor.accept(stream).await {
                    Ok(tls_stream) => {
                        info!("TLS handshake successful for {}", peer);
                        process_connection(
                            peer,
                            tls_stream,
                            agent_clone,
                            required_api_key
                        ).await
                    }
                    Err(e) => {
                        error!("TLS handshake error for {}: {}", peer, e);
                        Err(Box::new(e) as Box<dyn Error + Send + Sync>)
                    }
                }
            } else {
                process_connection(peer, stream, agent_clone, required_api_key).await
            };

            if let Err(e) = process_result {
                error!("Failed to process connection for {}: {}", peer, e);
            }
        });
    }
}

async fn process_connection<S>(
    peer: SocketAddr,
    stream: S,
    agent_clone: Arc<Mutex<AIAgent>>,
    required_api_key: Option<String>
) -> Result<(), Box<dyn Error + Send + Sync>>
    where S: AsyncRead + AsyncWrite + Unpin + Send + 'static
{
    let auth_callback = |req: &Request,  response: Response| -> Result<Response, ErrorResponse> {
        let secret = match &required_api_key {
            Some(k) if !k.is_empty() => k,
            _ => return Ok(response), 
        };

        let qs = req.uri().query().unwrap_or("");
        let params: HashMap<String, String> =
            form_urlencoded::parse(qs.as_bytes()).into_owned().collect();

        info!("Auth params from {}: {:?}", peer, params);

        let ts = params.get("ts")
            .or_else(|| params.get("X-Api-Ts"))
            .map(|s| s.as_str());
        let sig = params.get("sig")
            .or_else(|| params.get("X-Api-Sign")) 
            .map(|s| s.as_str());

        if let (Some(ts), Some(sig)) = (ts, sig) {
            let now = Utc::now().timestamp();
            let ts_i: i64 = ts.parse().unwrap_or(0);
            if (now - ts_i).abs() > 300 {
                let res = Response::builder()
                    .status(401) 
                    .body(Some("timestamp out of range".into()))
                    .unwrap();
                return Err(ErrorResponse::from(res));
            }

            let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
            mac.update(ts.as_bytes());
            let expected = hex::encode(mac.finalize().into_bytes());

            if expected == sig {
                Ok(response)
            } else {
                let res = Response::builder()
                    .status(401) 
                    .body(Some("bad signature".into()))
                    .unwrap();
                Err(ErrorResponse::from(res))
            }
        } else {
            let res = Response::builder()
                .status(401) 
                .body(Some("missing ts/sig".into()))
                .unwrap();
            Err(ErrorResponse::from(res))
        }
    };

    match accept_hdr_async(stream, auth_callback).await {
        Ok(ws) => {
            handle_connection(peer, ws, agent_clone).await;
            Ok(())
        }
        Err(e) => {
            error!("Handshake failed for {}: {}", peer, e);
            Err(Box::new(e) as _)
        }
    }
}

pub async fn handle_connection<S>(
    peer: SocketAddr,
    websocket: WebSocketStream<S>,
    agent: Arc<Mutex<AIAgent>>
)
    where S: AsyncRead + AsyncWrite + Unpin
{
    info!("New WebSocket connection: {}", peer);

    let (mut tx, mut rx) = websocket.split();
    let conversation_id = Uuid::new_v4().to_string();
    info!("Assigned conversation ID {} to {}", conversation_id, peer);

    let mut buffer = String::new();
    let mut in_thinking_section = false;
    let mut partial_close_tag = false;
    let mut partial_open_tag = false; // Add this for tracking partial opening tags

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
                            Ok(ClientMessage::Chat { content, capabilities }) => {
                                let client_supports_thinking = capabilities
                                    .as_ref()
                                    .map(|caps| caps.supports_thinking)
                                    .unwrap_or(false);

                                if client_supports_thinking {
                                    let thinking_start = ServerMessage::Thinking { 
                                        started: true,
                                    };
                                    let json = serde_json::to_string(&thinking_start).unwrap();
                                    if let Err(e) = tx.send(Message::Text(json)).await {
                                        error!("Error sending thinking start to {}: {}", peer, e);
                                    }
                                }

                                let typing_msg = ServerMessage::Typing;
                                if let Err(e) = tx.send(Message::Text(serde_json::to_string(&typing_msg).unwrap())).await {
                                    error!("Error sending typing status to {}: {}", peer, e);
                                    break;
                                }

                                let stream_result = agent
                                    .lock().await
                                    .process_message_stream(&conversation_id, &content)
                                    .await;

                                match stream_result {
                                    Ok(mut stream) => {
                                        while let Some(chunk_res) = stream.next().await {
                                            match chunk_res {
                                                Ok(fragment) => {
                                                    let text = fragment.as_str();
                                                    
                                                    // Check for split opening tag pattern
                                                    if !in_thinking_section && partial_open_tag && 
                                                       (text.starts_with(">") || text.starts_with("k>") || text.starts_with("nk>") || text.starts_with("ink>")) {
                                                        
                                                        partial_open_tag = false;
                                                        in_thinking_section = true;
                                                        
                                                        // Get everything after the ">" character
                                                        let after_tag_pos = text.find(">").unwrap_or(0) + 1;
                                                        let after_tag = &text[after_tag_pos..];
                                                        
                                                        // Send as thinking fragment
                                                        let msg = ServerMessage::ThinkingFragment { 
                                                            content: after_tag.to_string() 
                                                        };
                                                        tx.send(Message::Text(serde_json::to_string(&msg).unwrap())).await.unwrap();
                                                        
                                                        buffer = after_tag.to_string();
                                                        continue;
                                                    }
                                                    
                                                    buffer.push_str(text);
                                                    
                                                    // Check for potential partial opening tag at end of buffer
                                                    if !in_thinking_section && 
                                                       (buffer.ends_with("<t") || buffer.ends_with("<th") || 
                                                        buffer.ends_with("<thi") || buffer.ends_with("<thin") || 
                                                        buffer.ends_with("<think")) {
                                                        partial_open_tag = true;
                                                        continue;
                                                    }
                                                    
                                                    // Check for split closing tag pattern
                                                    if in_thinking_section && !buffer.contains("</think>") {
                                                        if buffer.ends_with("<") || (buffer.ends_with("</") && !text.starts_with("think>")) {
                                                            partial_close_tag = true;
                                                            continue;
                                                        }
                                                        
                                                        if partial_close_tag && text.starts_with("think>") || text.starts_with("/think>") {
                                                            in_thinking_section = false;
                                                            partial_close_tag = false;
                                                            
                                                            // Get content before the partial tag
                                                            let think_content = if buffer.ends_with("</") {
                                                                &buffer[..buffer.len()-2]
                                                            } else if buffer.ends_with("<") {
                                                                &buffer[..buffer.len()-1]
                                                            } else {
                                                                buffer.as_str()
                                                            };
                                                            
                                                            if !think_content.is_empty() {
                                                                let think_msg = ServerMessage::ThinkingFragment { 
                                                                    content: think_content.to_string() 
                                                                };
                                                                tx.send(Message::Text(serde_json::to_string(&think_msg).unwrap())).await.unwrap();
                                                            }
                                                            
                                                            // Extract and send anything after the closing tag
                                                            let after_tag_pos = text.find(">").unwrap_or(0) + 1;
                                                            if after_tag_pos < text.len() {
                                                                let after_content = &text[after_tag_pos..];
                                                                if !after_content.is_empty() {
                                                                    let clean_content = clean_response_text(after_content);
                                                                    let part = ServerMessage::Partial { content: clean_content };
                                                                    tx.send(Message::Text(serde_json::to_string(&part).unwrap())).await.unwrap();
                                                                }
                                                            }
                                                            
                                                            buffer.clear();
                                                            continue;
                                                        }
                                                    }
                                                    
                                                    // Regular tag processing (intact tags)
                                                    if !in_thinking_section && buffer.contains("<think>") {
                                                        in_thinking_section = true;
                                                        let start_pos = buffer.find("<think>").unwrap();
                                                        let after_tag = &buffer[start_pos + "<think>".len()..];
                                                        let msg = ServerMessage::ThinkingFragment { 
                                                            content: after_tag.to_string() 
                                                        };
                                                        tx.send(Message::Text(serde_json::to_string(&msg).unwrap())).await.unwrap();
                                                        
                                                        buffer = after_tag.to_string();
                                                        continue;
                                                    }
                                                    
                                                    if in_thinking_section && buffer.contains("</think>") {
                                                        let end_pos = buffer.find("</think>").unwrap();
                                                        let thinking_part = &buffer[..end_pos];
                                                        
                                                        if !thinking_part.is_empty() {
                                                            let think_msg = ServerMessage::ThinkingFragment { 
                                                                content: thinking_part.to_string() 
                                                            };
                                                            tx.send(Message::Text(serde_json::to_string(&think_msg).unwrap())).await.unwrap();
                                                        }
                                                        
                                                        in_thinking_section = false;
                                                        
                                                        // Extract everything after the closing tag
                                                        let after = buffer[end_pos + "</think>".len()..].to_string();
                                                        buffer.clear();
                                                        
                                                        // Process post-thinking content as regular partial content
                                                        if !after.is_empty() {
                                                            let clean_content = clean_response_text(&after);
                                                            let part = ServerMessage::Partial { 
                                                                content: clean_content 
                                                            };
                                                            tx.send(Message::Text(serde_json::to_string(&part).unwrap())).await.unwrap();
                                                        }
                                                        continue;
                                                    }
                                                    
                                                    // Flush buffer periodically to prevent buildup
                                                    if buffer.len() > 20 { 
                                                        if in_thinking_section {
                                                            let think_msg = ServerMessage::ThinkingFragment { 
                                                                content: buffer.clone() 
                                                            };
                                                            tx.send(Message::Text(serde_json::to_string(&think_msg).unwrap())).await.unwrap();
                                                        } else {
                                                            let clean_content = clean_response_text(&buffer);
                                                            let part = ServerMessage::Partial { 
                                                                content: clean_content 
                                                            };
                                                            tx.send(Message::Text(serde_json::to_string(&part).unwrap())).await.unwrap();
                                                        }
                                                        buffer.clear();
                                                    }
                                                }
                                                Err(e) => {
                                                    error!("Stream error for {}: {}", peer, e);
                                                    let error_msg = ServerMessage::Error {
                                                        message: format!("Stream error: {}", e),
                                                    };
                                                    let json = serde_json::to_string(&error_msg).unwrap();
                                                    if let Err(e_inner) = tx.send(Message::Text(json)).await {
                                                        error!("Error sending stream error to {}: {}", peer, e_inner);
                                                    }
                                                    break;
                                                }
                                            }
                                        }

                                        if !buffer.is_empty() {
                                            if in_thinking_section {
                                                let think_msg = ServerMessage::ThinkingFragment { content: buffer.clone() };
                                                tx.send(Message::Text(serde_json::to_string(&think_msg).unwrap())).await.unwrap();
                                            } else {
                                                let part = ServerMessage::Partial { content: buffer.clone() };
                                                tx.send(Message::Text(serde_json::to_string(&part).unwrap())).await.unwrap();
                                            }
                                        }

                                        let done_msg = ServerMessage::Done {
                                            timestamp: Utc::now().timestamp(),
                                        };
                                        let json = serde_json::to_string(&done_msg).unwrap();
                                        if let Err(e) = tx.send(Message::Text(json)).await {
                                            error!("Error sending done message to {}: {}", peer, e);
                                        }
                                    }
                                    Err(e) => {
                                        let error_message = format!("Error initiating stream: {}", e);
                                        error!("Agent streaming error for {}: {}", peer, error_message);
                                        let error_msg = ServerMessage::Error {
                                            message: error_message,
                                        };
                                        let json = serde_json::to_string(&error_msg).unwrap();
                                        if let Err(e_inner) = tx.send(Message::Text(json)).await {
                                            error!("Error sending error message to {}: {}", peer, e_inner);
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

async fn handle_message<S>(
    agent: Arc<Mutex<AIAgent>>,
    conversation_id: &str,
    message: &str,
    client_supports_thinking: bool,
    socket: &mut S
) -> Result<(), Box<dyn Error + Send + Sync>> 
where 
    S: SinkExt<Message> + Unpin,
    S::Error: std::fmt::Display,
{
    let thinking_response = agent.lock().await.process_message(conversation_id, message).await?;

    if !thinking_response.thinking.is_empty() {
        info!("LLM Thinking: {}", thinking_response.thinking);
    }

    let response_message = if client_supports_thinking {
        serde_json::json!({
            "response": thinking_response.response,
            "thinking": thinking_response.thinking
        }).to_string()
    } else {
        thinking_response.response
    };

    if let Err(e) = socket.send(Message::Text(response_message)).await {
        return Err(format!("Failed to send response: {}", e).into());
    }
    
    Ok(())
}

fn clean_response_text(text: &str) -> String {
    let mut cleaned = text.to_string();
    
    // Remove LaTeX formatting
    cleaned = cleaned.replace("\\boxed{", "").replace("\\text{", "");
    
    // Remove HTML/markdown formatting
    cleaned = cleaned.replace("\\<strong>", "").replace("\\</strong>", "")
                     .replace("**Final Answer:**", "")
                     .replace("**", "");
    
    // Remove common meta-commentary patterns
    let meta_patterns = [
        "The user's input is",
        "The appropriate response",
        "Final Answer:",
        "In response to",
        "I'll respond with"
    ];
    
    for pattern in &meta_patterns {
        if let Some(pos) = cleaned.find(pattern) {
            // Find the end of this meta-commentary section
            if let Some(end_pos) = cleaned[pos..].find("\n\n") {
                cleaned = cleaned[pos + end_pos + 2..].to_string();
            }
        }
    }
    
    // Clean up excessive whitespace and trim
    cleaned = cleaned.replace("\n\n\n", "\n\n").trim().to_string();
    
    cleaned
}