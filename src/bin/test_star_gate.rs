use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use env_logger;
use image::io::Reader as ImageReader;
use image::Rgb;
use imageproc::drawing;
use log::{info, warn};

use star_gate::algorithm::{bin_image, estimate_noise_from_image, get_stars_from_image};

/// Example program for running the StarGate star finding algorithm
/// on test image(s).
#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    /// Path of the file or directory to process.
    #[arg(short, long)]
    input: String,

    /// Directory where output file(s) are written.
    #[arg(short, long)]
    output: String,

    /// Statistical significance factor.
    #[arg(short, long, default_value_t = 8.0)]
    sigma: f32,

    /// Maximum star size.
    #[arg(short, long, default_value_t = 5)]
    max_size: u32,

    /// Whether image should be 2x2 binned prior to star detection.
    #[arg(short, long, default_value_t = false)]
    binning: bool,

    /// Output list of star centroids.
    #[arg(short, long, default_value_t = false)]
    coords: bool,
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
        for entry in fs::read_dir(&args.input).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                process_file(path.to_str().unwrap(), &args);
            }
        }
    } else {
        // Process the single file.
        assert!(input_metadata.is_file());
        process_file(args.input.as_str(), &args);
    }
}

fn process_file(file: &str, args: &Args) {
    info!("Processing {}", file);
    let input_path = PathBuf::from(&file);
    let mut output_path = PathBuf::from(&args.output);
    output_path.push(input_path.file_name().unwrap());
    output_path.set_extension("bmp");

    let img = match ImageReader::open(&input_path).unwrap().decode() {
        Ok(img) => img,
        Err(e) => {
            warn!("Skipping {:?} due to: {:?}", input_path, e);
            return;
        },
    };
    let mut img_u8 = img.to_luma8();
    let (width, height) = img_u8.dimensions();

    let star_extraction_start = Instant::now();
    let mut noise_estimate = estimate_noise_from_image(&img_u8);
    if args.binning {
        img_u8 = bin_image(&img_u8, noise_estimate, args.sigma);
        noise_estimate = estimate_noise_from_image(&img_u8);
    }
    let (mut stars, _hot_pixel_count, _binned_image) =
        get_stars_from_image(&img_u8, noise_estimate, args.sigma, args.max_size,
                             /*detect_hot_pixels=*/!args.binning,
                             /*create_binned_image=*/false);
    let elapsed = star_extraction_start.elapsed();
    info!("WxH: {}x{}; noise level {}", width, height, noise_estimate);
    info!("Star extraction found {} stars in {:?}", stars.len(), elapsed);
    info!("{}ms per megapixel\n",
          elapsed.as_secs_f32() * 1000.0 / ((width * height) as f32 / 1000000.0));

    // Sort by brightness estimate, brightest first.
    stars.sort_by(|a, b| b.mean_brightness.partial_cmp(&a.mean_brightness).unwrap());

    // Scribble marks into the image showing where we found stars.
    let mut img_color = img.into_rgb8();
    let coord_mul = if args.binning { 2.0 } else { 1.0 };
    for (index, star) in stars.iter().enumerate() {
        // Stars early in list (bright) get brighter circle.
        let progress = index as f64 / stars.len() as f64;
        let circle_brightness = 100 + ((1.0 - progress) * 155.0) as u8;
        drawing::draw_hollow_circle_mut(
            &mut img_color,
            ((star.centroid_x * coord_mul) as i32,
             (star.centroid_y * coord_mul) as i32),
            4,
            Rgb::<u8>([circle_brightness, 0, 0]));
    }
    img_color.save(output_path).unwrap();
    if args.coords {
        let mut coords_str = String::new();
        coords_str.push_str(format!("# WxH {}x{}\n", width, height).as_str());
        coords_str.push_str("# (y, x)\n");
        for star in stars {
            coords_str.push_str(format!(
                "({}, {}),\n",
                star.centroid_y * coord_mul, star.centroid_x * coord_mul).as_str());
        }
        info!("{}", coords_str);
    }
}
