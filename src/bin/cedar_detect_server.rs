// Copyright (c) 2023 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

use std::ffi::CString;
use std::io::Error;
use std::net::SocketAddr;
use std::time::Instant;

use clap::Parser;
use env_logger;
use image::{GrayImage};
use imageproc::rect::Rect;
use libc::{c_void, close, mmap, munmap, shm_open, O_RDONLY, PROT_READ, MAP_FAILED, MAP_SHARED};
use log::{debug, info, warn};

use ::cedar_detect::algorithm::{estimate_noise_from_image,
                                estimate_background_from_image_region,
                                get_stars_from_image};
use crate::cedar_detect::cedar_detect_server::{CedarDetect, CedarDetectServer};

use tonic_web::GrpcWebLayer;

pub mod cedar_detect {
    // The string specified here must match the proto package name.
    tonic::include_proto!("cedar_detect");
}

struct MyCedarDetect {
    // No server state; pure function calls.
}

#[tonic::async_trait]
impl CedarDetect for MyCedarDetect {
    async fn extract_centroids(
        &self, request: tonic::Request<cedar_detect::CentroidsRequest>)
        -> Result<tonic::Response<cedar_detect::CentroidsResult>, tonic::Status>
    {
        let rpc_start = Instant::now();
        let req: cedar_detect::CentroidsRequest = request.into_inner();

        if req.input_image.is_none() {
            return Err(tonic::Status::invalid_argument(
                "Request 'input_image' field is missing"));
        }
        let input_image = req.input_image.unwrap();

        let req_image;
        let mut fd = 0;
        let mut addr: *mut c_void = std::ptr::null_mut();
        let mut num_pixels = 0;
        let using_shmem = input_image.shmem_name.is_some();
        if using_shmem {
            num_pixels = (input_image.width * input_image.height) as usize;
            let name = CString::new(input_image.shmem_name.unwrap()).unwrap();
            debug!("Using shared memory at {:?}", name);
            unsafe {
                fd = shm_open(name.as_ptr(), O_RDONLY, 0);
                if fd < 0 {
                    let msg = format!(
                        "Could not open shared memory at {:?}: errno {}",
                        name, Error::last_os_error().raw_os_error().unwrap());
                    warn!("{}", msg);
                    // Clever client-side logic can recognize the INTERNAL error and
                    // fall back to not using shared memory.
                    return Err(tonic::Status::internal(msg))
                }
                addr = mmap(std::ptr::null_mut(), num_pixels, PROT_READ,
                            MAP_SHARED, fd, 0);
                if addr == MAP_FAILED {
                    let msg = format!(
                        "Could not mmap shared memory at {:?} for {} bytes: errno {}",
                        name, num_pixels, Error::last_os_error().raw_os_error().unwrap());
                    warn!("{}", msg);
                    if close(fd) == -1 {
                        warn!("Could not close shared memory file: errno {}",
                              Error::last_os_error().raw_os_error().unwrap());
                    }
                    return Err(tonic::Status::internal(msg))
                }
                // We are violating the invariant that 'addr' must have been
                // allocated using the global allocator. This is OK because our
                // usage of the vector (within GrayImage) is read-only, so there
                // won't be any reallocations. Note that we call
                // vec_shmem.leak() below, so the vector won't try to deallocate
                // 'addr'.
                let vec_shmem = Vec::<u8>::from_raw_parts(addr as *mut u8,
                                                          num_pixels, num_pixels);
                req_image = GrayImage::from_raw(input_image.width as u32,
                                                input_image.height as u32,
                                                vec_shmem).unwrap();
            }
        } else {
            req_image = GrayImage::from_raw(input_image.width as u32,
                                            input_image.height as u32,
                                            input_image.image_data).unwrap();
        }

        let noise_estimate = estimate_noise_from_image(&req_image);
        let (stars, hot_pixel_count, binned_image, _histogram) = get_stars_from_image(
            &req_image, noise_estimate, req.sigma, req.max_size as u32,
            req.use_binned_for_star_candidates, req.detect_hot_pixels,
            req.return_binned);

        let mut background_estimate: Option<f32> = None;
        if let Some(estimate_background_region) = req.estimate_background_region {
            if estimate_background_region.origin_x < 0 ||
                estimate_background_region.origin_y < 0 ||
                estimate_background_region.origin_x +
                estimate_background_region.width > input_image.width ||
                estimate_background_region.origin_y +
                estimate_background_region.height > input_image.height
            {
                return Err(tonic::Status::invalid_argument(format!(
                    "Invalid estimate_background_region {:?}",
                    estimate_background_region)));
            }
            background_estimate = Some(estimate_background_from_image_region(
                &req_image,
                &Rect::at(estimate_background_region.origin_x,
                          estimate_background_region.origin_y)
                    .of_size(estimate_background_region.width as u32,
                             estimate_background_region.height as u32)));
        }

        if using_shmem {
            // Deconstruct req_image that is referencing shared memory.
            let vec_shmem = req_image.into_raw();
            vec_shmem.leak();  // vec_shmem no longer "owns" the shared memory.
            unsafe {
                if munmap(addr, num_pixels) == -1 {
                    warn!("Could not munmap shared memory: errno {}",
                          Error::last_os_error().raw_os_error().unwrap());
                }
                if close(fd) == -1 {
                    warn!("Could not close shared memory file: errno {}",
                          Error::last_os_error().raw_os_error().unwrap());
                }
            }
        }

        // Average the peak pixels of the N brightest stars.
        let mut sum_peak: i32 = 0;
        let mut num_peak = 0;
        const NUM_PEAKS: i32 = 10;

        let mut candidates = Vec::<cedar_detect::StarCentroid>::new();
        for star in stars {
            if num_peak < NUM_PEAKS {
                sum_peak += star.peak_value as i32;
                num_peak += 1;
            }
            candidates.push(cedar_detect::StarCentroid{
                centroid_position: Some(cedar_detect::ImageCoord{
                    x: star.centroid_x,
                    y: star.centroid_y,
                }),
                brightness: star.brightness,
                num_saturated: star.num_saturated as i32,
            });
        }
        let response = cedar_detect::CentroidsResult{
            noise_estimate,
            background_estimate,
            hot_pixel_count: hot_pixel_count as i32,
            peak_star_pixel: if num_peak > 0 { sum_peak / num_peak } else { 255 },
            star_candidates: candidates,
            binned_image: if binned_image.is_some() {
                let bimg: GrayImage = binned_image.unwrap();
                Some(cedar_detect::Image {
                    width: bimg.width() as i32,
                    height: bimg.height() as i32,
                    image_data: bimg.into_raw(),
                    shmem_name: None,
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    /// Port that the gRPC server listens on.
    #[arg(short, long, default_value_t = 50051)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    // Listen on any address for the given port.
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    info!("CedarDetectServer listening on {}", addr);

    tonic::transport::Server::builder()
        .accept_http1(true)
        .layer(GrpcWebLayer::new())
        .add_service(CedarDetectServer::new(MyCedarDetect{}))
        .serve(addr)
        .await?;
    Ok(())
}
