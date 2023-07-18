use std::env;

use env_logger;
use image::io::Reader as ImageReader;
use image::Luma;

use star_gate::get_centroids_from_image;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = env::args().collect();

    let image_file = &args[1];
    let output_file = &args[2];
    let img = ImageReader::open(image_file).unwrap().decode().unwrap();
    let mut img_u8 = img.into_luma8();
    let (width, height) = img_u8.dimensions();

    let sigma = 6.0;
    let mut candidates = get_centroids_from_image(&img_u8, sigma);
    candidates.sort_by(|a, b| b.sum.partial_cmp(&a.sum).unwrap());

    // Scribble marks into the image showing where we found stars.
    for candidate in candidates {
        let x = candidate.centroid_x as u32;
        let y = candidate.centroid_y as u32;
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
    img_u8.save(output_file).unwrap();
}
