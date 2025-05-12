use serde::{ Serialize, Deserialize };

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "chat")]
    Chat { 
        content: String,
        #[serde(default)]
        capabilities: Option<ClientCapabilities>
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ClientCapabilities {
    #[serde(default)]
    pub supports_thinking: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "response")]
    Response { content: String },
    
    #[serde(rename = "partial")]
    Partial { content: String },
    
    #[serde(rename = "thinking")]
    Thinking { started: bool },
    
    #[serde(rename = "thinking_fragment")]
    ThinkingFragment { content: String },
    
    #[serde(rename = "error")]
    Error { message: String },
    
    #[serde(rename = "typing")]
    Typing,
    
    #[serde(rename = "done")]
    Done { timestamp: i64 },
}
