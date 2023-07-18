use std::cmp;
use std::collections::hash_map::HashMap;
use std::time::Instant;

use image::GrayImage;
use log::{debug, info};

fn estimate_noise_from_image(image: &GrayImage) -> f32 {
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

#[derive(Copy, Clone, Debug)]
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
    for rownum in 3..height-3 {
        // Get the slice of image_pixels corresponding to this row.
        let row_pixels: &[u8] = &image_pixels.as_slice()
            [(rownum * width) as usize .. ((rownum+1) * width) as usize];
        // Slide a 7 pixel window across the row.
        let mut center_x = 2;
        for window in row_pixels.windows(7) {
            center_x += 1;
            let lb = window[0] as i16;  // Left border.
            let ll = window[1] as i16;  // Second left neighbor.
            let l = window[2] as i16;   // Left neighbor.
            let c = window[3] as i16;   // Center.
            let r = window[4] as i16;   // Right neighbor.
            let rr = window[5] as i16;  // Second right neighbor.
            let rb = window[6] as i16;  // Right border.
            // Center pixel must be at least as bright as its immediate left/right
            // neighbors.
            if l > c || c < r {
                continue;
            }
            // Center pixel must be strictly brighter than its second left/right
            // neighbors.
            if ll >= c || c <= rr {
                continue;
            }
            // Center pixel must be strictly brighter than the borders.
            if lb >= c || c <= rb {
                continue;
            }
            if l == c {
                // Break tie between left and center.
                if ll > r {
                    // Left will have been the center of its own candidate entry.
                    continue;
                }
            }
            if c == r {
                // Break tie between center and right.
                if l <= rr {
                    // Right will be the center of its own candidate entry.
                    continue;
                }
            }
            // We require the border pixels to be ~uniformly dark. See if there
            // is too much brightness difference between the border pixels.
            let border_diff = (lb - rb).abs();
            if border_diff as f64 > 0.5 * sigma as f64 * noise_estimate as f64 {
                continue;
            }
            // Center pixel must be sigma * estimated noise brighter than
            // the estimated background.
            let center_2 = 2 * c;
            let est_background_2 = lb + rb;
            let est_noise_2 = 2.0 * noise_estimate;
            let center_over_background_2 = center_2 - est_background_2;
            if center_over_background_2 < (sigma * est_noise_2 as f32) as i16 {
                continue;
            }
            // Average of l+r must be 0.5 * sigma * estimated noise brighter
            // than the estimated background.
            let sum_neighbors_over_background = l + r - est_background_2;
            if sum_neighbors_over_background <
                (sigma * noise_estimate as f32) as i16 {
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

#[derive(Debug)]
struct Blob {
    candidates: Vec<CandidateFromRowScan>,

    // If candidates is empty, that means this blob has been merged into
    // another blob.
    recipient_blob: i32,
}

#[derive(Copy, Clone)]
struct LabeledCandidate {
    candidate: CandidateFromRowScan,
    blob_id: i32,
}

fn form_blobs_from_candidates(candidates: Vec<CandidateFromRowScan>, height: i32)
                              -> Vec<Blob> {
    let blobs_start = Instant::now();
    let mut labeled_candidates_by_row: Vec<Vec<LabeledCandidate>> = Vec::new();
    labeled_candidates_by_row.resize(height as usize, Vec::<LabeledCandidate>::new());

    let mut blobs: HashMap<i32, Blob> = HashMap::new();
    let mut next_blob_id = 0;
    // Create an initial singular blob for each candidate.
    for candidate in candidates {
        blobs.insert(next_blob_id, Blob{candidates: vec![candidate],
                                        recipient_blob: -1});
        labeled_candidates_by_row[candidate.y as usize].push(
            LabeledCandidate{candidate, blob_id: next_blob_id});
        next_blob_id += 1;
    }

    // TODO: blobs can be adjacent within a row
    // TODO: broaden the overlap width.

    // Merge adjacent blobs. Within a row blobs are not adjacent (by definition of
    // how row scanning identifies candidates), so we just need to look for vertical
    // adjacencies.
    // Start processing at row 1 so we can look to previous row for blob merges.
    for rownum in 1..height as usize {
        for rc in &labeled_candidates_by_row[rownum] {
            let rc_pos = rc.candidate.x;
            // See if rc is adjacent to any candidates in the previous row.
            // This is fast since rows usually have very few candidates.
            for prev_row_rc in &labeled_candidates_by_row[rownum - 1] {
                let prev_row_rc_pos = prev_row_rc.candidate.x;
                if prev_row_rc_pos <= rc_pos - 3 {
                    continue;
                }
                if prev_row_rc_pos >= rc_pos + 3 {
                    break;
                }
                // Adjacent to a candidate in the previous row. Absorb the previous
                // row blob's candidates.
                let recipient_blob_id = rc.blob_id;
                let mut donor_blob_id = prev_row_rc.blob_id;
                let mut donated_candidates: Vec<CandidateFromRowScan>;
                loop {
                    let donor_blob = blobs.get_mut(&donor_blob_id).unwrap();
                    if !donor_blob.candidates.is_empty() {
                        donated_candidates = donor_blob.candidates.drain(..).collect();
                        assert!(donor_blob.recipient_blob == -1);
                        donor_blob.recipient_blob = recipient_blob_id;
                        break;
                    }
                    // This blob got merged to another blob.
                    assert!(donor_blob.recipient_blob != -1);
                    donor_blob_id = donor_blob.recipient_blob;
                }
                let recipient_blob = &mut blobs.get_mut(&recipient_blob_id).unwrap();
                recipient_blob.candidates.append(&mut donated_candidates);
            }
        }
    }
    // Return non-empty blobs. Note that the blob merging we just did will leave
    // some empty entries in the `blobs` mapping.
    let mut non_empty_blobs = Vec::<Blob>::new();
    for (_id, blob) in blobs {
        if !blob.candidates.is_empty() {
            debug!("got blob {:?}", blob);
            non_empty_blobs.push(blob);
        }
    }
    info!("Found {} blobs in {:?}", non_empty_blobs.len(), blobs_start.elapsed());
    non_empty_blobs
}

#[derive(Debug)]
pub struct StarDescription {
    // Location of star centroid in image coordinates. (0.5, 0.5) corresponds
    // to the center of the image's upper left pixel.
    pub centroid_x: f32,
    pub centroid_y: f32,

    // Characterizes the extent or spread of the star in each direction, in
    // pixel units.
    pub stddev_x: f32,
    pub stddev_y: f32,

    // TODO: other moments for elongation and angle.

    // Sum of the u8 pixel values of the star's region. The estimated background
    // is subtracted before the summing.
    pub sum: f32,

    // TODO: count of saturated pixels.
}

fn get_star_from_blob(blob: &Blob, image: &GrayImage,
                      noise_estimate: f32, sigma: f32,
                      max_width: u32, max_height: u32) -> Option<StarDescription> {
    let mut x_min = u32::MAX;
    let mut x_max = 0_u32;
    let mut y_min = u32::MAX;
    let mut y_max = 0_u32;
    for candidate in &blob.candidates {
        x_min = cmp::min(x_min, candidate.x as u32);
        x_max = cmp::max(x_max, candidate.x as u32);
        y_min = cmp::min(y_min, candidate.y as u32);
        y_max = cmp::max(y_max, candidate.y as u32);
    }
    let blob_width: u32 = x_max - x_min + 1;
    let blob_height: u32 = y_max - y_min + 1;
    // Reject blob if it is too big.
    if blob_width > max_width || blob_height > max_height {
        debug!("Blob too large at WxH {}x{}", blob_width, blob_height);
        return None;
    }
    // Expand box by a three pixel border in all directions.
    x_min -= 3;
    x_max += 3;
    y_min -= 3;
    y_max += 3;
    debug!("blob x range: {}-{}", x_min, x_max);
    debug!("blob y range: {}-{}", y_min, y_max);
    // for row in y_min..y_max+1 {
    //     let mut values = Vec::<String>::new();
    //     for col in x_min..x_max+1 {
    //         values.push(format!("{}", image.get_pixel(col as u32, row as u32).0[0]));
    //     }
    //     debug!("{} ", values.join(" "));
    // }

    // Gather the pixel values from the outer perimeter. These are used to
    // estimate the background.
    let mut perimeter_pixels = Vec::<f64>::new();
    for x in x_min..x_max+1 {
        perimeter_pixels.push(image.get_pixel(x, y_min).0[0] as f64);
        perimeter_pixels.push(image.get_pixel(x, y_max).0[0] as f64);
    }
    for y in y_min+1..y_max {
        perimeter_pixels.push(image.get_pixel(x_min, y).0[0] as f64);
        perimeter_pixels.push(image.get_pixel(x_max, y).0[0] as f64);
    }
    debug!("perimeter: {:?} ", perimeter_pixels);
    let background_est: f64 = perimeter_pixels.iter().sum::<f64>() /
        perimeter_pixels.len() as f64;
    debug!("background: {} ", background_est);

    // We require the perimeter pixels to be ~uniformly dark. See if any
    // perimeter pixel is too bright compared to the darkest perimeter
    // pixel.
    let perimeter_min =
        perimeter_pixels.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let perimeter_max =
        perimeter_pixels.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    if perimeter_max - perimeter_min > sigma as f64 * noise_estimate as f64 {
        debug!("Border too varied");
        return None;
    }

    // Process the interior pixels to obtain moments.
    let mut m0: f64 = 0.0;
    let mut m1x: f64 = 0.0;
    let mut m1y: f64 = 0.0;
    for y in y_min+2..y_max-1 {
        for x in x_min+2..x_max-1 {
            let val = image.get_pixel(x, y).0[0] as f64 - background_est;
            m0 += val;
            m1x += x as f64 * val;
            m1y += y as f64 * val;
        }
    }
    // See if the integrated background adjusted brightness exceeds the
    // sigma*noise by enough.
    if m0 < sigma as f64 * noise_estimate as f64 {
        debug!("Blob too weak");
        return None;
    }
    // We use simple center-of-mass as the centroid.
    let centroid_x = m1x / m0;
    let centroid_y = m1y / m0;
    // Compute second moment about the centroid.
    let mut m2x_c: f64 = 0.0;
    let mut m2y_c: f64 = 0.0;
    for y in y_min+2..y_max-1 {
        for x in x_min+2..x_max-1 {
            let val = image.get_pixel(x, y).0[0] as f64 - background_est;
            m2x_c += (x as f64 - centroid_x) * (x as f64 - centroid_x) * val;
            m2y_c += (y as f64 - centroid_y) * (y as f64 - centroid_y) * val;
        }
    }
    let variance_x = m2x_c / m0;
    let variance_y = m2y_c / m0;
    debug!("centroid x,y {},{}; variance x,y {},{}",
           centroid_x, centroid_y, variance_x, variance_y);

    Some(StarDescription{centroid_x: (centroid_x + 0.5) as f32,
                         centroid_y: (centroid_y + 0.5) as f32,
                         stddev_x: variance_x.sqrt() as f32,
                         stddev_y: variance_y.sqrt() as f32,
                         sum: m0 as f32})
}

pub fn get_centroids_from_image(image: &GrayImage, sigma: f32)
                                -> Vec<StarDescription> {
    let (width, height) = image.dimensions();
    info!("Image width x height: {}x{}", width, height);

    let mut noise_estimate = estimate_noise_from_image(image);
    if noise_estimate < 1.0 {
        // Likely the image background is crushed to black.
        noise_estimate = 1.0;
    }
    let candidates = scan_rows_for_candidates(image, noise_estimate, sigma);
    let blobs = form_blobs_from_candidates(candidates, height as i32);

    let get_stars_start = Instant::now();
    let mut stars: Vec<StarDescription> = Vec::new();
    for blob in blobs {
        match get_star_from_blob(&blob, image, noise_estimate, sigma, 8, 8) {
            Some(x) => stars.push(x),
            None => ()
        }
    }
    info!("Blob processing found {} stars in {:?}",
          stars.len(), get_stars_start.elapsed());
    stars
}
