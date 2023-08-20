use std::net::SocketAddr;
use std::time::{Duration, Instant};

use clap::Parser;
use env_logger;
use log::{info, warn};

use ::star_gate::algorithm::{bin_image, estimate_noise_from_image, get_stars_from_image};

use crate::star_gate::star_gate_server::{StarGate, StarGateServer};
use crate::star_gate::{CentroidsRequest, CentroidsResult, StarCentroid};

use tonic_web::GrpcWebLayer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    /// Port that the gRPC server listens on.
    #[arg(short, long, default_value_t = 50051)]
    port: u16,
}

pub mod star_gate {
    // The string specified here must match the proto package name.
    tonic::include_proto!("star_gate");
}

struct MyStarGate {
    // No server state; pure function calls.
}

#[tonic::async_trait]
impl StarGate for MyStarGate {
    async fn extract_centroids(
        &self, request: tonic::Request<CentroidsRequest>)
        -> Result<tonic::Response<CentroidsResult>, tonic::Status>
    {
        let req: CentroidsRequest = request.into_inner();

        let candidates = Vec::<StarCentroid>::new();
        let response = star_gate::CentroidsResult{
            noise_estimate: 0.0,
            hot_pixel_count: 0,
            star_candidates: candidates,
            binned_image: None,
            algorithm_time: Some(prost_types::Duration::try_from(Duration::from_secs(0)).unwrap()),
        };
        Ok(tonic::Response::new(response))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    // Listen on any address for the given port.
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    info!("StarGateServer listening on {}", addr);

    tonic::transport::Server::builder()
        .accept_http1(true)
        .layer(GrpcWebLayer::new())
        .add_service(StarGateServer::new(MyStarGate{}))
        .serve(addr)
        .await?;
    Ok(())
}
