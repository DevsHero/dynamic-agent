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
use tokio_tungstenite::tungstenite::handshake::server::{ Request, Response };
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::tungstenite::http::response::Response as HttpResponse;
use tokio_rustls::TlsAcceptor;
use rustls::ServerConfig;
use rustls::pki_types::{ CertificateDer, PrivateKeyDer };
use rustls_pemfile::{ certs, pkcs8_private_keys };
use lazy_static::lazy_static;
use governor::{ RateLimiter, Quota, state::{ InMemoryState, NotKeyed }, clock::DefaultClock };

use log::{ info, warn, error, debug };

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
        let auth_callback = |
            req: &Request,
            response: Response
        | -> Result<Response, HttpResponse<Option<String>>> {
            info!("Handshake from {}", peer);

            let mut provided = req
                .headers()
                .get("X-API-Key")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned);

            if provided.is_none() {
                if let Some(q) = req.uri().query() {
                    for pair in q.split('&') {
                        let mut kv = pair.splitn(2, '=');
                        if kv.next() == Some("api_key") {
                            provided = kv.next().map(|v| v.to_string());
                            break;
                        }
                    }
                }
            }

            debug!("Client provided API key: {:?}", provided);

            if let Some(ref required) = required_api_key {
                if provided.as_deref() != Some(required.as_str()) {
                    warn!("{}: bad or missing API key", peer);
                    let resp = HttpResponse::builder()
                        .status(StatusCode::UNAUTHORIZED)
                        .header("Content-Type", "text/plain")
                        .body(Some("Unauthorized".into()))
                        .unwrap();
                    return Err(resp);
                }
                info!("{} authenticated", peer);
            } else {
                info!("{} no API key required", peer);
            }

            Ok(response)
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
