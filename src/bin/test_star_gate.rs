use std::env;
use std::time::Instant;

use env_logger;
use image::io::Reader as ImageReader;
use image::Luma;
use log::{info};

use star_gate::get_stars_from_image;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = env::args().collect();

    let image_file = &args[1];
    let output_file = &args[2];
    let img = ImageReader::open(image_file).unwrap().decode().unwrap();
    let mut img_u8 = img.into_luma8();
    let (width, height) = img_u8.dimensions();
    info!("Image width x height: {}x{}", width, height);

    let star_extraction_start = Instant::now();
    let return_candidates = false;
    let (stars, _hot_pixel_count, _noise_estimate) =
        get_stars_from_image(&img_u8, /*sigma=*/6.0, return_candidates);
    let elapsed = star_extraction_start.elapsed();
    info!("Star extraction found {} stars in {:?}", stars.len(), elapsed);
    info!("{}ms per megapixel",
          elapsed.as_secs_f32() * 1000.0 / ((width * height) as f32 / 1000000_f32));

    // Scribble marks into the image showing where we found stars.
    for star in stars {
        let x = star.centroid_x as u32;
        let y = star.centroid_y as u32;
        let grey = Luma::<u8>([127]);
        if x > 6 {
            // Draw left tick.
            img_u8.put_pixel(x - 4, y, grey);
            img_u8.put_pixel(x - 5, y, grey);
            img_u8.put_pixel(x - 6, y, grey);
        }
        if x < width - 6 {
            // Draw right tick.
            img_u8.put_pixel(x + 4, y, grey);
            img_u8.put_pixel(x + 5, y, grey);
            img_u8.put_pixel(x + 6, y, grey);
        }
        if !return_candidates {
            if y > 6 {
                // Draw top tick.
                img_u8.put_pixel(x, y - 4, grey);
                img_u8.put_pixel(x, y - 5, grey);
                img_u8.put_pixel(x, y - 6, grey);
            }
            if y < height - 6 {
                // Draw bottom tick.
                img_u8.put_pixel(x, y + 4, grey);
                img_u8.put_pixel(x, y + 5, grey);
                img_u8.put_pixel(x, y + 6, grey);
            }
        }
    }
    img_u8.save(output_file).unwrap();
}
