use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use env_logger;
use image::io::Reader as ImageReader;
use image::Luma;
use imageproc::drawing;
use log::{info, warn};

use star_gate::get_stars_from_image;

/// Example program for running the StarGate star finding algorithm
/// on test image(s).
#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    /// Name of the file or directory to process.
    #[arg(short, long)]
    input: String,

    /// Directory where output file(s) are written.
    #[arg(short, long)]
    output: String,

    /// Statistical significance factor.
    #[arg(short, long, default_value_t = 6.0)]
    sigma: f32,

    /// Maximum star size.
    #[arg(short, long, default_value_t = 5)]
    max_size: u32,
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let input_metadata = fs::metadata(&args.input).unwrap_or_else(|e| {
        panic!("Input file/dir '{}' does not exist? {:?}", args.input, e);
    });
    let output_metadata = fs::metadata(&args.output).unwrap_or_else(|e| {
        panic!("Output dir '{}' does not exist? {:?}", args.output, e);
    });
    assert!(output_metadata.is_dir(),
            "Output '{}' must be a directory", args.output);
    if input_metadata.is_dir() {
        // Enumerate and process all of the files in the directory.
        for entry in fs::read_dir(args.input).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                process_file(path.to_str().unwrap(), &args.output, args.sigma, args.max_size);
            }
        }
    } else {
        assert!(input_metadata.is_file());
        process_file(args.input.as_str(), args.output.as_str(), args.sigma, args.max_size);
    }
}

fn process_file(file: &str, output_dir: &str, sigma: f32, max_size: u32) {
    info!("Processing {}", file);
    let input_path = PathBuf::from(&file);
    let mut output_path = PathBuf::from(output_dir);
    output_path.push(input_path.file_name().unwrap());
    output_path.set_extension("bmp");

    let img = match ImageReader::open(&input_path).unwrap().decode() {
        Ok(img) => img,
        Err(e) => {
            warn!("Skipping {:?} due to: {:?}", input_path, e);
            return;
        },
    };
    let mut img_u8 = img.into_luma8();
    let (width, height) = img_u8.dimensions();

    let star_extraction_start = Instant::now();
    let (mut stars, _hot_pixel_count, noise_estimate) =
        get_stars_from_image(&img_u8, sigma, max_size);
    let elapsed = star_extraction_start.elapsed();
    info!("WxH: {}x{}; noise level {}", width, height, noise_estimate);
    info!("Star extraction found {} stars in {:?}", stars.len(), elapsed);
    info!("{}ms per megapixel\n",
          elapsed.as_secs_f32() * 1000.0 / ((width * height) as f32 / 1000000.0));

    // Sort by brightness estimate, brightest first.
    stars.sort_by(|a, b| b.mean_brightness.partial_cmp(&a.mean_brightness).unwrap());

    // Scribble marks into the image showing where we found stars.
    for (index, star) in stars.iter().enumerate() {
        // Stars early in list (bright) get brighter circle.
        let progress = index as f64 / stars.len() as f64;
        let circle_brightness = 30 + ((1.0 - progress) * 100.0) as u8;
        drawing::draw_hollow_circle_mut(
            &mut img_u8,
            (star.centroid_x as i32, star.centroid_y as i32),
            4,
            Luma::<u8>([circle_brightness]));
    }
    img_u8.save(output_path).unwrap();
}
