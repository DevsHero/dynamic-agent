use crate::agent::AIAgent;
use crate::websocket::handle_connection;
use crate::cli::Args;
use std::error::Error;
use std::sync::Arc;
use std::fs::File;
use std::io::BufReader;
use std::num::NonZeroU32;
use std::net::SocketAddr;
use tokio::sync::Mutex;
use tokio::net::TcpListener;
use tokio::io::{ AsyncRead, AsyncWrite };
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::handshake::server::{ Request, Response, ErrorResponse };
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::tungstenite::http::response::Response as HttpResponse;
use tokio_rustls::TlsAcceptor;
use rustls::ServerConfig;
use rustls::pki_types::{ CertificateDer, PrivateKeyDer };
use rustls_pemfile::{ certs, pkcs8_private_keys };
use lazy_static::lazy_static;
use governor::{ RateLimiter, Quota, state::{ InMemoryState, NotKeyed }, clock::DefaultClock };
use hmac::{Hmac, Mac};
use sha2::Sha256;
use chrono::Utc;
use hex;
use std::collections::HashMap;
use url::form_urlencoded;

type HmacSha256 = Hmac<Sha256>;

use log::{ info, warn, error  };

lazy_static! {
    static ref CONNECTION_LIMITER: RateLimiter<NotKeyed, InMemoryState, DefaultClock> = RateLimiter::direct(Quota::per_second(NonZeroU32::new(10).unwrap()));
}

pub struct Server {
    addr: String,
    agent: Arc<Mutex<AIAgent>>,
    api_key: Option<String>,
    args: Args,
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

impl Server {
    pub fn new(addr: String, agent: Arc<AIAgent>, api_key: Option<String>, args: Args) -> Self {
        let inner_agent = (*agent).clone();
        let agent_arc = Arc::new(Mutex::new(inner_agent));
        let api_key = api_key.filter(|k| !k.trim().is_empty());

        if api_key.is_some() {
            info!("Server configured with API Key authentication.");
        } else {
            warn!("Server configured WITHOUT API Key authentication. Connections are open.");
        }

        Self { addr, agent: agent_arc, api_key, args }
    }

    pub async fn run(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(&self.addr).await?;

        let protocol = if
            self.args.enable_tls &&
            self.args.tls_cert_path.is_some() &&
            self.args.tls_key_path.is_some()
        {
            "wss"
        } else {
            "ws"
        };
        info!("{} server listening on: {}", protocol.to_uppercase(), self.addr);

        let tls_acceptor = if self.args.enable_tls {
            match (&self.args.tls_cert_path, &self.args.tls_key_path) {
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
                    error!(
                        "Both --tls-cert-path and --tls-key-path must be provided to enable TLS."
                    );
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
            let agent_clone = Arc::clone(&self.agent);
            let required_api_key = self.api_key.clone();
            let tls_acceptor_clone = tls_acceptor.clone();

            tokio::spawn(async move {
                let process_result = if let Some(acceptor) = tls_acceptor_clone {
                    match acceptor.accept(stream).await {
                        Ok(tls_stream) => {
                            info!("TLS handshake successful for {}", peer);
                            Self::process_connection(
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
                    Self::process_connection(peer, stream, agent_clone, required_api_key).await
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
        let auth_callback = |req: &Request, mut response: Response| -> Result<Response, ErrorResponse> {
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
                    let mut res = Response::builder()
                        .status(StatusCode::UNAUTHORIZED)
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
                    let mut res = Response::builder()
                        .status(StatusCode::UNAUTHORIZED)
                        .body(Some("bad signature".into()))
                        .unwrap();
                    Err(ErrorResponse::from(res))
                }
            } else {
                let  res = Response::builder()
                    .status(StatusCode::UNAUTHORIZED)
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
}
