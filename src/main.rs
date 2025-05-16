use clap::Parser;
use dynamic_agent::cli::Args;
use dotenv::dotenv;
use rustls::crypto::ring;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {

    ring::default_provider()
        .install_default()
        .expect("crypto provider already installed");

    dotenv().ok();
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let args = Args::parse();
    dynamic_agent::run(args).await
}
