use std::net::SocketAddr;
use std::time::Instant;

use clap::Parser;
use env_logger;
use image::GrayImage;
use log::{info};

use ::star_gate::algorithm::{bin_image, estimate_noise_from_image, get_stars_from_image};
use crate::star_gate::star_gate_server::{StarGate, StarGateServer};

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
        &self, request: tonic::Request<star_gate::CentroidsRequest>)
        -> Result<tonic::Response<star_gate::CentroidsResult>, tonic::Status>
    {
        let rpc_start = Instant::now();
        let req: star_gate::CentroidsRequest = request.into_inner();

        if req.input_image.is_none() {
            return Err(tonic::Status::invalid_argument(
                "Request 'input_image' field is missing"));
        }
        let input_image = req.input_image.unwrap();

        let mut req_image = GrayImage::from_raw(input_image.width as u32,
                                                input_image.height as u32,
                                                input_image.image_data).unwrap();
        let mut noise_estimate = estimate_noise_from_image(&req_image);
        if req.use_binned_for_star_candidates {
            req_image = bin_image(&req_image, noise_estimate, req.sigma);
            noise_estimate = estimate_noise_from_image(&req_image);
        }
        let (mut stars, hot_pixel_count, binned_image) =
            get_stars_from_image(
                &req_image, noise_estimate, req.sigma, req.max_size as u32,
                /*detect_hot_pixels=*/!req.use_binned_for_star_candidates,
                /*create_binned_image=*/req.return_binned);
        // Sort by brightness estimate, brightest first.
        stars.sort_by(|a, b| b.mean_brightness.partial_cmp(&a.mean_brightness).unwrap());

        let coord_mul = if req.use_binned_for_star_candidates { 2.0 } else { 1.0 };
        let mut candidates = Vec::<star_gate::StarCentroid>::new();
        for star in stars {
            candidates.push(star_gate::StarCentroid{
                centroid_position: Some(star_gate::ImageCoord{
                    x: star.centroid_x * coord_mul,
                    y: star.centroid_y * coord_mul,
                }),
                stddev_x: star.stddev_x * coord_mul,
                stddev_y: star.stddev_y * coord_mul,
                mean_brightness: star.mean_brightness,
                background: star.background,
                num_saturated: star.num_saturated as i32,
            });
        }
        let response = star_gate::CentroidsResult{
            noise_estimate,
            hot_pixel_count: hot_pixel_count as i32,
            star_candidates: candidates,
            binned_image: if binned_image.is_some() {
                let bimg: GrayImage = binned_image.unwrap();
                Some(star_gate::Image {
                    width: bimg.width() as i32,
                    height: bimg.height() as i32,
                    image_data: bimg.into_raw(),
                })
            } else {
                None
            },
            algorithm_time: Some(prost_types::Duration::try_from(
                rpc_start.elapsed()).unwrap()),
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
