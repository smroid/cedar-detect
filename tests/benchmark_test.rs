// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use image::ImageReader;
use cedar_detect::algorithm::{estimate_noise_from_image, get_stars_from_image};

#[test]
fn benchmark_detection_all_images() {
    let test_data_dir = "test_data";

    let mut img_files: Vec<PathBuf> = fs::read_dir(test_data_dir)
        .unwrap_or_else(|e| panic!("Failed to read directory '{}': {:?}", test_data_dir, e))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && matches!(
            path.extension().and_then(|e| e.to_str()), Some("jpg") | Some("bmp")))
        .collect();
    img_files.sort();

    // Algorithm parameters matching the defaults from test_cedar_detect.rs
    let sigma = 8.0;
    let normalize_rows = false;
    let binning = 2;
    let hot_pixels = true;
    let iterations = 1000;

    println!("============================================================");
    println!("Starting benchmark for {} iterations per image...", iterations);
    println!("============================================================\n");

    let mut grand_total_time = Duration::new(0, 0);
    let mut grand_total_stars = 0usize;
    let mut files_processed = 0;

    for input_path in img_files {
        let file_name = input_path.file_name().unwrap().to_string_lossy();

        // Load and decode the image OUTSIDE the timed loop to ensure we
        // strictly measure only the detection algorithm's execution time.
        let img = match ImageReader::open(&input_path) {
            Ok(reader) => match reader.decode() {
                Ok(decoded) => decoded,
                Err(e) => {
                    println!("Failed to decode image '{}': {:?}", file_name, e);
                    continue;
                }
            },
            Err(e) => {
                println!("Failed to open image '{}': {:?}", file_name, e);
                continue;
            }
        };

        let img_u8 = img.into_luma8();
        let mut total_detection_time = Duration::new(0, 0);
        let mut expected_stars: Option<usize> = None;

        println!("Benchmarking: {}", file_name);

        for _ in 0..iterations {
            let start = Instant::now();

            // Noise estimation is typically a prerequisite step for the core detection pipeline
            let noise_estimate = estimate_noise_from_image(&img_u8);

            // Core star detection algorithm
            let (stars, _, _, _) = get_stars_from_image(
                &img_u8,
                noise_estimate,
                sigma,
                normalize_rows,
                binning,
                hot_pixels,
                false // return_binned_image
            );

            // Accumulate only the execution time
            total_detection_time += start.elapsed();

            let num_stars = stars.len();
            match expected_stars {
                None => expected_stars = Some(num_stars),
                Some(expected) => assert_eq!(num_stars, expected,
                    "Star count changed between iterations for '{}'", file_name),
            }
        }

        grand_total_time += total_detection_time;
        grand_total_stars += expected_stars.unwrap_or(0);
        files_processed += 1;

        println!("  Stars Detected:         {}", expected_stars.unwrap_or(0));
        println!("  Total Detection Time:   {:?}", total_detection_time);
        println!("  Average Time/Iteration: {:?}", total_detection_time / iterations as u32);
        println!("------------------------------------------------------------");
    }

    if files_processed > 0 {
        let total_iterations = iterations * files_processed;
        println!("\n============================================================");
        println!("GRAND TOTALS ({} files, {} iterations each)", files_processed, iterations);
        println!("============================================================");
        println!("Overall Total Stars Detected:        {}", grand_total_stars);
        println!("Overall Total Detection Time:       {:?}", grand_total_time);
        println!("Average Time Per File ({} runs):  {:?}", iterations, grand_total_time / files_processed as u32);
        println!("Average Time Per Single Detection:  {:?}", grand_total_time / total_iterations as u32);
        println!("============================================================");
    }
}
