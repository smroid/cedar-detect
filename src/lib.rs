use std::collections::hash_map::HashMap;
use std::time::Instant;

use image::GrayImage;
use log::{debug, info};

pub fn estimate_noise_from_image(image: &GrayImage) -> f32 {
    let noise_start = Instant::now();
    let (width, height) = image.dimensions();
    let box_size = 20;
    if width < box_size || height < box_size {
        panic!("Image is too small WxH {}x{}", width, height);
    }
    // TODO: probe a couple of areas of the image and pick the one with lower
    // overall values. Goal: avoid doing noise stats on e.g. the moon.
    let start_x = width / 2 - box_size / 2;
    let start_y = height / 2 - box_size / 2;
    let mut histogram: [i32; 256] = [0; 256];
    for y in start_y..start_y+box_size {
        for x in start_x..start_x+box_size {
            let pixel_value = image.get_pixel(x, y);
            histogram[pixel_value.0[0] as usize] += 1;
        }
    }
    debug!("Histogram: {:?}", histogram);
    // Discard the top 5% of the histogram. We want only non-star pixels
    // to contribute to the noise estimate.
    let keep_count: i32 = (box_size * box_size * 95 / 100) as i32;
    let mut kept_so_far = 0;
    let mut first_moment = 0;
    for h in 0..256 {
        let bin_count = histogram[h];
        if kept_so_far + bin_count > keep_count {
            histogram[h] = keep_count - kept_so_far;
        }
        kept_so_far += histogram[h];
        first_moment += histogram[h] * h as i32;
    }
    let mean = first_moment as f64 / keep_count as f64;
    let mut second_moment: f64 = 0.0;
    for h in 0..256 {
        second_moment += histogram[h] as f64 * (h as f64 - mean) * (h as f64 - mean);
    }
    let stddev: f64 = (second_moment / keep_count as f64).sqrt();
    info!("Noise estimate {} found in {:?}", stddev, noise_start.elapsed());
    stddev as f32
}

#[derive(Copy, Clone)]
struct CandidateFromRowScan {
    x: i32,
    y: i32,
}

// The candidates are returned in raster scan order.
fn scan_rows_for_candidates(image: &GrayImage, noise_estimate: f32, sigma: f32)
                            -> Vec<CandidateFromRowScan> {
    let row_scan_start = Instant::now();
    let (width, height) = image.dimensions();
    let image_pixels: &Vec<u8> = image.as_raw();
    let mut candidates: Vec<CandidateFromRowScan> = Vec::new();

    // TODO: shard the rows to multiple threads.
    for rownum in 0..height {
        // Get the slice of image_pixels corresponding to this row.
        let row_pixels: &[u8] = &image_pixels.as_slice()
            [(rownum * width) as usize .. ((rownum+1) * width) as usize];
        // Slide a 5 pixel window across the row.
        let mut center_x = 1;
        for window in row_pixels.windows(5) {
            center_x += 1;
            let left_border = window[0] as i16;
            let left = window[1] as i16;
            let center = window[2] as i16;
            let right = window[3] as i16;
            let right_border = window[4] as i16;
            // Center pixel must be at least as bright as its immediate left/right
            // neighbors.
            if left > center || center < right {
                continue;
            }
            // Center pixel must be strictly brighter than both border pixels.
            if left_border >= center || center <= right_border {
                continue;
            }
            if left == center {
                // Break tie between left and center.
                if left_border > right {
                    // Left will have been the center of its own candidate entry.
                    continue;
                }
            }
            if right == center {
                // Break tie between center and right.
                if left <= right_border {
                    // Right will be the center of its own candidate entry.
                    continue;
                }
            }
            // Center pixel must be sigma * estimated noise brighter than
            // the estimated background.
            let center_2 = 2 * center;
            let est_background_2 = left_border + right_border;
            let est_noise_2 = 2.0 * noise_estimate;
            let center_over_background_2 = center_2 - est_background_2;
            if center_over_background_2 < (sigma * est_noise_2 as f32) as i16 {
                continue;
            }
            // Sum of left+right pixels must be sigma * estimated noise
            // brighter than the estimated background.
            let neighbors_over_background_2 = left + right - est_background_2;
            if neighbors_over_background_2 < (sigma * est_noise_2 as f32) as i16 {
                continue;
            }
            // To guard against being at a bright spot in an extended object such
            // as the moon, we require the center pixel to not just be
            // arithmetically above the background, but also to be a ratio higher
            // than the left_border and right_border background pixels.
            let frac_center = center * 3 / 4;
            if frac_center < left_border || frac_center < right_border {
                continue;
            }

            // We have a candidate star from our 1d analysis!
            candidates.push(CandidateFromRowScan{x: center_x, y: rownum as i32});
            debug!("Candidate at row {} col {}; window {:?}", rownum, center_x, window);
        }
    }

    info!("Row scan found {} candidates in {:?}",
          candidates.len(), row_scan_start.elapsed());
    candidates
}

pub struct StarDescription {
    // Location of star centroid in image coordinates. (0.5, 0.5) corresponds
    // to the center of the image's upper left pixel.
    pub centroid_x: f32,
    pub centroid_y: f32,

    // Sum of the u8 pixel values of the star's region. The estimated background
    // is subtracted before the summing.
    pub sum: i32,
}

#[derive(Copy, Clone)]
struct LabeledCandidate {
    candidate: CandidateFromRowScan,
    blob_id: i32,
}

struct Blob {
    candidates: Vec<LabeledCandidate>,
}

pub fn get_centroids_from_image(image: &GrayImage, sigma: f32)
                                -> Vec<StarDescription> {
    let get_centroids_start = Instant::now();
    let (width, height) = image.dimensions();
    info!("Image width x height: {}x{}", width, height);

    let mut noise_estimate = estimate_noise_from_image(image);
    if noise_estimate == 0.0 {
        // Likely the image background is crushed to black.
        noise_estimate = 1.0;
    }
    let candidates = scan_rows_for_candidates(image, noise_estimate, sigma);

    let mut labeled_candidates_by_row: Vec<Vec<LabeledCandidate>> = Vec::new();
    labeled_candidates_by_row.resize(height as usize, Vec::<LabeledCandidate>::new());

    let mut blobs: HashMap<i32, Blob> = HashMap::new();
    let mut next_blob_id = 0;
    // Create an initial singular Blob for each candidate.
    for candidate in candidates {
        let labeled_candidate = LabeledCandidate{candidate, blob_id: next_blob_id};
        blobs.insert(next_blob_id, Blob{candidates: vec![labeled_candidate]});
        labeled_candidates_by_row[candidate.y as usize].push(labeled_candidate);
        next_blob_id += 1;
    }




    let mut results: Vec<StarDescription> = Vec::new();

    info!("Found {} centroids in {:?}",
          results.len(), get_centroids_start.elapsed());
    results
}
