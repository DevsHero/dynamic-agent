pub mod api;
pub mod websocket;

use crate::agent::AIAgent;
use crate::cli::Args;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Mutex;
 

pub struct Server {
    addr: String,
    agent: Arc<Mutex<AIAgent>>,
    args: Args,
  
}

impl Server {
    pub fn new(
        addr: String,
        agent: Arc<Mutex<AIAgent>>,
        args: Args, 
    ) -> Self {
        Self {
            addr,
            agent,
            args,
           
        }
    }

    pub async fn run(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if let Some(http_port) = self.args.http_port {
            self.start_http_server(http_port).await?;
        }
        
        self.start_ws_server().await?;
        
        Ok(())
    }
    
    async fn start_http_server(&self, http_port: u16) -> Result<(), Box<dyn Error + Send + Sync>> {
        api::start_http_server(
            http_port,
            self.agent.clone(),
            self.args.clone(),
        ).await
    }
    
    async fn start_ws_server(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        websocket::start_ws_server(
            &self.addr,
            self.agent.clone(),
            None,
            self.args.clone(),
        ).await
    }
}