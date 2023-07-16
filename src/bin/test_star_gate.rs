use std::env;

use env_logger;
use image::io::Reader as ImageReader;
use star_gate::get_centroids_from_image;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = env::args().collect();

    let image_file = &args[1];
    let img = ImageReader::open(image_file).unwrap().decode().unwrap();
    let img_u8 = img.into_luma8();

    let candidates = get_centroids_from_image(&img_u8, 5.0);
}
