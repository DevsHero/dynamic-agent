use serde::{ Serialize, Deserialize };

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "chat")] Chat {
        content: String,
    },
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "response")] Response {
        content: String,
        timestamp: i64,
    },
    #[serde(rename = "error")] Error {
        message: String,
    },
    #[serde(rename = "processing")]
    Processing,
}
