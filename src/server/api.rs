use crate::agent::AIAgent;
use crate::cli::Args;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use axum::{
    routing::get,
    Router,
    extract::{State, Query},
    response::IntoResponse,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use log::{info, error};

#[derive(Deserialize)]
pub struct ReloadRequest {
    pub source: Option<String>, 
}

#[derive(Serialize)]
struct ReloadResponse {
    success: bool,
    message: String,
    details: Option<Vec<String>>,
}

#[derive(Clone)]
struct AppState {
    agent: Arc<Mutex<AIAgent>>,
    args: Args,
}

pub async fn start_http_server(
    http_port: u16,
    agent: Arc<Mutex<AIAgent>>,
    args: Args,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let addr = format!("0.0.0.0:{}", http_port).parse::<SocketAddr>()?;
    info!("Starting HTTP API server on: http://{}", addr);

    let app_state = AppState {
        agent,
        args: args.clone(),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/reload-prompts", get(reload_prompts_handler))
        .layer(cors)
        .with_state(app_state);

    if args.enable_tls && args.tls_cert_path.is_some() && args.tls_key_path.is_some() {
        let cert_path = args.tls_cert_path.as_ref().unwrap();
        let key_path = args.tls_key_path.as_ref().unwrap();
        
        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(
            cert_path,
            key_path
        ).await?;

        tokio::spawn(async move {
            let result = axum_server::bind_rustls(addr, tls_config)
                .serve(app.into_make_service())
                .await;

            if let Err(e) = result {
                error!("HTTPS server error: {}", e);
            }
        });

        info!("HTTPS server started with TLS enabled");
    } else {

        tokio::spawn(async move {
            match tokio::net::TcpListener::bind(addr).await {
                Ok(listener) => {
                    if let Err(e) = axum::serve(listener, app.into_make_service()).await {
                        error!("HTTP server error: {}", e);
                    }
                },
                Err(e) => {
                    error!("Failed to bind HTTP server to {}: {}. Try a different port.", addr, e);
                }
            }
        });

        info!("HTTP server started");
    }

    Ok(())
}

async fn reload_prompts_handler(
    State(state): State<AppState>,
    Query(req): Query<ReloadRequest>,
) -> impl IntoResponse {
    let mut agent = match state.agent.try_lock() {
        Ok(g) => g,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(ReloadResponse {
            success: false,
            message: "Agent busy".into(),
            details: None,
        })).into_response(),
    };

    let source = req.source.as_deref().unwrap_or("local");
    let mut results = Vec::new();
    let mut ok = true;

    match source {
        "local" => {
            match agent.reload_prompts_if_changed(&state.args).await {
                Ok(true) => results.push("Local reloaded".into()),
                Ok(false) => results.push("Local unchanged".into()),
                Err(e) => { ok = false; results.push(format!("Local error: {}", e)); }
            }
        }
        "remote" => {
            if !state.args.enable_remote_prompts {
                results.push("Remote disabled".into());
            } else {
                match agent.force_refresh_remote_prompts(&state.args).await {
                    Ok(true) => results.push("Remote reloaded".into()),
                    Ok(false) => results.push("Remote unchanged".into()),
                    Err(e) => { ok = false; results.push(format!("Remote error: {}", e)); }
                }
            }
        }
        _ => { 
            match agent.reload_prompts_if_changed(&state.args).await {
                Ok(true) => results.push("Local reloaded".into()),
                Ok(false) => results.push("Local unchanged".into()),
                Err(e) => { ok = false; results.push(format!("Local error: {}", e)); }
            }
            if state.args.enable_remote_prompts {
                match agent.force_refresh_remote_prompts(&state.args).await {
                    Ok(true) => results.push("Remote reloaded".into()),
                    Ok(false) => results.push("Remote unchanged".into()),
                    Err(e) => { ok = false; results.push(format!("Remote error: {}", e)); }
                }
            } else {
                results.push("Remote disabled".into());
            }
        }
    }

    let code = if ok { StatusCode::OK } else { StatusCode::BAD_REQUEST };
    (code, axum::Json(ReloadResponse {
        success: ok,
        message: if ok { "Reload complete".into() } else { "Reload errors".into() },
        details: Some(results),
    })).into_response()
}