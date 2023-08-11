//! StarGate provides efficient and accurate detection of stars in sky images.
//! Given an image, StarGate returns a list of detected star centroids expressed
//! in image pixel coordinates.
//!
//! Features:
//!
//! * Employs localized thresholding to tolerate changes in background levels
//!   across the image.
//! * Adapts to different image exposure levels.
//! * Estimates noise in the image and adapts the star detection threshold
//!   accordingly.
//! * Automatically classifies and rejects hot pixels.
//! * Rejects trailed objects such as aircraft lights or satellites.
//! * Tolerates the presence of bright interlopers such as the moon or
//!   streetlights.
//! * Simple function call interface with few parameters aside from the input
//!   image.
//! * Fast! On a Raspberry Pi 4B, the execution time per 1M image pixels is
//!   around 5ms, even when several dozen stars are present in the image.
//!
//! # Intended applications
//!
//! StarGate is designed to be used with astrometry and plate solving systems
//! such as [Tetra3](https://github.com/esa/tetra3). It can also be incorporated
//! into satellite star trackers.
//!
//! A goal of StarGate is to allow such applications to achieve fast response
//! times. StarGate contributes to this by running quickly and by tolerating a
//! degree of image noise allowing for shorter imaging integration times.
//!
//! # Star detection fidelity
//!
//! Like any detection algorithm, StarGate produces both false negatives (failures
//! to detect actual stars) and false positives (spurious star detections). The
//! Caveats section below mentions some causes of false negatives.
//!
//! False positives can occur as you reduce the `sigma` parameter when calling
//! [get_stars_from_image()] in an attempt to increase the number of detected
//! stars. Although StarGate requires evidence from multiple pixels to qualify
//! each star detection (reducing the occurrence of false positives), noise
//! inevitably wins if you push `sigma` too low.
//!
//! # Algorithm
//!
//! StarGate spends most of its time on a single efficient pass over the input
//! image. This generates a set of star candidates that are subjected to further
//! scrutiny. The candidates number in the hundreds or perhaps thousands (thus
//! much less than the number of image pixels), so the candidate evaluation
//! usually takes only a fraction of the overall time.
//!
//! The code comments have much more detail.
//!
//! # Caveats
//!
//! ## Star extraction only
//!
//! StarGate is designed to identify only star-like spots. It is not a
//! generalized astronomical source extraction system.
//!
//! ## Crowding
//!
//! The criteria used by StarGate to efficiently detect stars are designed
//! around the characteristics of a star image's pixels compared to surrounding
//! pixels. StarGate can thus be confused when stars are too closely spaced, or
//! a star is close to a hot pixel. Such situations will usually cause closely
//! spaced stars to fail to be detected. Note that for applications such as
//! plate solving, this is probably for the better. A star that is rejected
//! because it is near a hot pixel is a rare misfortune that we accept.
//!
//! ## Imaging requirements
//!
//! * StarGate supports only 8-bit grayscale images; color images or images with
//!   greater bit depth must be converted before calling
//!   [get_stars_from_image()].
//! * The imaging exposure time and sensor gain are usually chosen by the caller
//!   to yield a desired number of faint star detections. In so doing, if a
//!   bright star is overexposed to a degree that it bleeds into too many
//!   adjacent pixels, StarGate will reject the bright star.
//! * Pixel scale and focusing are crucial:
//!   * If star images are too extended w.r.t. the pixel grid, StarGate will not
//!     detect the stars. You'll either need to tighten the focus or use pixel
//!     binning before calling [get_stars_from_image()].
//!   * If pixels are too large w.r.t. the stars' images, StarGate will
//!     mis-classify many stars as hot pixels. A simple remedy is to slightly
//!     defocus such that stars occupy a central peak pixel with a bit of spread
//!     into immediately adjacent pixels.
//! * StarGate does not tolerate more than maybe a single pixel of motion blur.
//!
//! ## Centroid estimation
//!
//! StarGate's computes a sub-pixel centroid position for each detected star
//! using the first moment of pixel intensity (center of gravity) calculated
//! over a bounding box around the star. This suffices for many purposes, but a
//! more exacting application might want to do its own centroiding:
//!
//! * If a higher bit-depth image was converted to 8 bit for calling
//!   [get_stars_from_image()], centroiding in the original image may yield
//!   better accuracy.
//! * The application can use a more artful centroiding technique such as
//!   gaussian fitting.
//!
//! ## Lens distortions
//!
//! Optical imaging systems can produce distortions that affect StarGate in two
//! ways:
//!
//! * Optical aberrations can cause star images to deviate from ideal small
//!   condensed spots. StarGate will fail to detect stars that are too distorted,
//!   such as near the corners of a wide field image.
//! * Lens pincushion and/or barrel distortions can cause reported star positions
//!   to deviate from their true positions. StarGate reports star centroids in
//!   the image's x/y coordinates; it is up to the application to apply any needed
//!   lens distortion corrections.

use std::cmp;
use std::collections::hash_map::HashMap;
use std::time::Instant;

use image::GrayImage;
use imageproc::rect::Rect;
use log::{debug, info};

// An iterator over the pixels of a region of interest. Yields pixels in raster
// scan order.
struct EnumeratePixels<'a> {
    image: &'a GrayImage,
    roi: &'a Rect,
    include_interior: bool,

    // Identifies the next pixel to be yielded. If cur_y is beyond the ROI's
    // bottom, the iteration is finished.
    cur_x: i32,
    cur_y: i32,
}

impl<'a> EnumeratePixels<'a> {
    // If include_interior is false, only the perimeter is enumerated.
    fn new(image: &'a GrayImage, roi: &'a Rect, include_interior: bool)
           -> EnumeratePixels<'a> {
        let (width, height) = image.dimensions();
        assert!(roi.left() >= 0);
        assert!(roi.top() >= 0);
        assert!(roi.right() < width as i32);
        assert!(roi.bottom() < height as i32);
        EnumeratePixels{image, roi, include_interior,
                        cur_x: roi.left(), cur_y: roi.top()}
    }
}

impl<'a> Iterator for EnumeratePixels<'a> {
    type Item = (i32, i32, u8);  // x, y, pixel value.

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_y > self.roi.bottom() {
            return None;
        }
        let item:Self::Item = (self.cur_x, self.cur_y,
                               self.image.get_pixel(
                                   self.cur_x as u32, self.cur_y as u32).0[0]);
        if self.cur_x == self.roi.right() {
            self.cur_x = self.roi.left();
            self.cur_y += 1;
        } else {
            let do_all_in_row = self.include_interior ||
                self.cur_y == self.roi.top() || self.cur_y == self.roi.bottom();
            if do_all_in_row {
                self.cur_x += 1;
            } else {
                // Exclude interior.
                assert!(self.cur_x == self.roi.left());
                self.cur_x = self.roi.right();
            }
        }
        Some(item)
    }
}

// Given a candidate pixel, examines that pixel within a 7 pixel horizontal
// context (the candidate pixel plus three neighbors on each side).
//
// We label the pixels as: |lb lm l C r rm rb| where:
// lb: left border
// lm: left margin
// l:  left neighbor
// C:  center or candidate pixel
// r:  right neighbor
// rm: right margin
// rb: right border
//
// When the candidate pixel is the center pixel of a horizontal cut through a
// star, we will observe/require the following:
//
// * The center pixel will be the brightest.
// * The outermost border pixels are taken to reflect the local sky background
//   value.
// * The center pixel will be brighter than the sky background by a
//   statistically significant amount.
// * The left+right neighbors might not be as bright as the center pixel,
//   but they will be brighter than some fraction of the center pixel. This
//   criterion lets us reject single hot pixels.
//
// If a 7 pixel horizontal window satisfies these (and other; see the code)
// criteria, the center pixel is flagged as being a candidate for futher
// analysis. If any of these criteria are not met, the center pixel is deemed to
// be NOT a star.
//
// Note that the left and right "margin" pixels don't figure into the above
// narration; a 5-pixel horizontal window might have been used instead. The
// margin pixels allow a slightly higher degree of star defocus compared to what
// a 5 pixel window would permit.
//
// Statistical significance is defined as a 'sigma' multiple of the measured
// image noise level. The `sigma_noise_2` argument is 2x the sigma*noise value,
// and the `sigma_noise_1_5` argument is 1.5x the sigma*noise value.
//
// Returns:
//   0: Corrected pixel value for use in summarize_region_of_interest(). The
//      value is NOT background subtracted, but care is taken to substitute hot
//      pixel value with its neighboring pixel value.
//   1: Whether the gate's center pixel is a star candidate, a hot pixel, or
//      uninteresting.
#[derive(Debug, Eq, PartialEq)]
enum ResultType {
    Uninteresting,
    Candidate,
    HotPixel,
}
fn gate_star_1d(gate: &[u8], sigma_noise_2: i16, sigma_noise_1_5: i16)
                -> (/*corrected_value*/u8, ResultType) {
    debug_assert!(sigma_noise_2 > 0);
    debug_assert!(sigma_noise_1_5 > 0);
    debug_assert!(sigma_noise_1_5 <= sigma_noise_2);
    // Examining assembler output suggests that fetching these exterior pixels
    // first eliminates bounds checks on the interior pixels. I would have
    // thought that the compiler would do this... in any case, the measured
    // performance doesn't seem to change.
    let lb = gate[0] as i16;
    let rb = gate[6] as i16;

    let lm = gate[1] as i16;
    let l = gate[2] as i16;
    let c = gate[3] as i16;
    let r = gate[4] as i16;
    let rm = gate[5] as i16;
    let c8 = gate[3];

    // Center pixel must be sigma * estimated noise brighter than the estimated
    // background. Do this test first, because it eliminates the vast majority
    // of candidates.
    let est_background_2 = lb + rb;
    let center_minus_background_2 = c + c - est_background_2;
    if center_minus_background_2 < sigma_noise_2 {
        return (c8, ResultType::Uninteresting);
    }
    // Center pixel must be at least as bright as its immediate left/right
    // neighbors.
    if l > c || c < r {
        return (c8, ResultType::Uninteresting);
    }
    // Center pixel must be strictly brighter than its left/right margin.
    if lm >= c || c <= rm {
        return (c8, ResultType::Uninteresting);
    }
    if l == c {
        // Break tie between left and center.
        if lm > r {
            // Left will have been the center of its own candidate entry.
            return (c8, ResultType::Uninteresting);
        }
    }
    if c == r {
        // Break tie between center and right.
        if l <= rm {
            // Right will be the center of its own candidate entry.
            return (c8, ResultType::Uninteresting);
        }
    }
    // Average of l+r (minus background) must exceed 0.25 * center (minus
    // background).
    let sum_neighbors_minus_background = l + r - est_background_2;
    if 4 * sum_neighbors_minus_background <= center_minus_background_2 {
        // For ROI processing purposes, replace the hot pixel with its
        // neighbors' value.
        return (((l + r) / 2) as u8, ResultType::HotPixel);
    }
    // We require the border pixels to be ~uniformly dark. See if there is too
    // much brightness difference between the border pixels.
    // The 3/2 sigma_noise threshold is empirically chosen to yield a low
    // rejection rate for actual sky background border pixels.
    let border_diff = (lb - rb).abs();
    if border_diff > sigma_noise_1_5 {
        return (c8, ResultType::Uninteresting);
    }
    // We have a candidate star from our 1d analysis!
    debug!("candidate: {:?}", gate);
    return (c8, ResultType::Candidate);
}

#[derive(Copy, Clone, Debug)]
struct CandidateFrom1D {
    x: i32,
    y: i32,
}

// Applies gate_star_1d() at all pixel positions of the image (excluding the few
// leftmost and rightmost columns) to identify star candidates for futher
// screening.
// `image` Image to be scanned.
// `noise_estimate` The noise level of the image. See estimate_noise_from_image().
// `sigma` Specifies the multiple of the noise level by which a pixel must exceed
//     the background to be considered a star candidate, in addition to
//     satisfying other criteria.
// Returns:
// Vec<CandidateFrom1D>: the identifed star candidates, in raster scan order.
// u32: count of hot pixels detected.
fn scan_image_for_candidates(image: &GrayImage, noise_estimate: f32, sigma: f32)
                       -> (Vec<CandidateFrom1D>, /*hot_pixel_count*/u32) {
    let row_scan_start = Instant::now();
    let mut hot_pixel_count = 0_u32;
    let (width, height) = image.dimensions();
    let image_pixels: &Vec<u8> = image.as_raw();
    let mut candidates = Vec::<CandidateFrom1D>::new();
    let sigma_noise_2 = cmp::max((2.0 * sigma * noise_estimate + 0.5) as i16, 2);
    let sigma_noise_1_5 = cmp::max((1.5 * sigma * noise_estimate + 0.5) as i16, 1);
    // We'll generally have way fewer than 1 candidate per row.
    candidates.reserve(height as usize);
    for rownum in 0..height {
        // Get the slice of image_pixels corresponding to this row.
        let row_pixels: &[u8] = &image_pixels.as_slice()
            [(rownum * width) as usize .. ((rownum+1) * width) as usize];
        // Slide a 7 pixel gate across the row.
        let mut center_x = 2_u32;
        for gate in row_pixels.windows(7) {
            center_x += 1;
            let (_pixel_value, result_type) =
                gate_star_1d(gate, sigma_noise_2, sigma_noise_1_5);
            match result_type {
                ResultType::Uninteresting => (),
                ResultType::Candidate => {
                    debug!("Candidate at row {} col {}; gate {:?}",
                           rownum, center_x, gate);
                    candidates.push(CandidateFrom1D{x: center_x as i32,
                                                    y: rownum as i32});
                },
                ResultType::HotPixel => {
                    debug!("Hot pixel at row {} col {}; gate {:?}",
                           rownum, center_x, gate);
                    hot_pixel_count += 1;
                },
            }
        }
    }
    info!("Image scan found {} candidates and {} hot pixels in {:?}",
          candidates.len(), hot_pixel_count, row_scan_start.elapsed());
    (candidates, hot_pixel_count)
}

#[derive(Debug)]
struct Blob {
    candidates: Vec<CandidateFrom1D>,

    // If candidates is empty, that means this blob has been merged into
    // another blob.
    recipient_blob: i32,
}

#[derive(Copy, Clone)]
struct LabeledCandidate {
    candidate: CandidateFrom1D,
    blob_id: i32,
}

// The scan_image_for_candidates() function can produce multiple candidates for
// the same star. This typically happens for a bright star that has some
// vertical extent such that multiple rows' horizontal cuts through the star
// all flag it as a candidate.
//
// The form_blobs_from_candidates() function gathers connected candidates into
// blobs which will be further analyzed to determine if they are stars.
fn form_blobs_from_candidates(candidates: Vec<CandidateFrom1D>)
                              -> Vec<Blob> {
    let blobs_start = Instant::now();
    let mut labeled_candidates_by_row = Vec::<Vec<LabeledCandidate>>::new();

    let mut blobs: HashMap<i32, Blob> = HashMap::new();
    let mut next_blob_id = 0;
    // Create an initial singular blob for each candidate.
    for candidate in candidates {
        blobs.insert(next_blob_id, Blob{candidates: vec![candidate],
                                        recipient_blob: -1});
        if candidate.y as usize >= labeled_candidates_by_row.len() {
            labeled_candidates_by_row.resize(candidate.y as usize + 1,
                                             Vec::<LabeledCandidate>::new());
        }
        labeled_candidates_by_row[candidate.y as usize].push(
            LabeledCandidate{candidate, blob_id: next_blob_id});
        next_blob_id += 1;
    }
    // Merge adjacent blobs. Within a row blobs are not adjacent (by the nature of
    // how row scanning identifies candidates), so we just need to look for vertical
    // adjacencies.
    // Start processing at row 1 so we can look to previous row for blob merges.
    for rownum in 1..labeled_candidates_by_row.len() {
        for rc in &labeled_candidates_by_row[rownum] {
            let rc_pos = rc.candidate.x;
            // See if rc is adjacent to any candidates in the previous row.
            // This is fast since rows usually have very few candidates.
            for prev_row_rc in &labeled_candidates_by_row[rownum - 1] {
                let prev_row_rc_pos = prev_row_rc.candidate.x;
                if prev_row_rc_pos < rc_pos - 3 {
                    continue;
                }
                if prev_row_rc_pos > rc_pos + 3 {
                    break;
                }
                // Adjacent to a candidate in the previous row. Absorb the previous
                // row blob's candidates.
                let recipient_blob_id = rc.blob_id;
                let mut donor_blob_id = prev_row_rc.blob_id;
                let mut donated_candidates: Vec<CandidateFrom1D>;
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
            assert!(blob.recipient_blob == -1);
            debug!("got blob {:?}", blob);
            non_empty_blobs.push(blob);
        }
    }
    info!("Found {} blobs in {:?}", non_empty_blobs.len(), blobs_start.elapsed());
    non_empty_blobs
}

/// Summarizes a star-like spot found by [get_stars_from_image()].
#[derive(Debug)]
pub struct StarDescription {
    /// Location of star centroid in image coordinates. (0.5, 0.5) corresponds
    /// to the center of the image's upper left pixel.
    pub centroid_x: f32,
    pub centroid_y: f32,

    /// Characterizes the extent or spread of the star in each direction, in
    /// pixel size units.
    pub stddev_x: f32,
    pub stddev_y: f32,

    /// Mean of the u8 pixel values of the star's region (core plus immediate
    /// neighbors). The estimated background is subtracted.
    pub mean_brightness: f32,

    /// The estimated sky background near the star.
    pub background: f32,

    /// Count of saturated pixel values.
    pub num_saturated: u16,
}

// The gate_star_2d() function is the 2-D version of gate_star_1d(). While the
// latter is simplistic because it is applied to every single image pixel, the
// gate_star_2d() function can be more thorough because it is only applied to
// hundreds or maybe thousands of candidates.
//
// A Blob is one or more candidates from gate_star_1d(), grouped together
// by form_blobs_from_candidates(). We define the following terms:
//
// Core: this is the bounding box that includes all of the input Blob's pixel
//   coordinates. In most cases the core consists of a single pixel, but for
//   brighter stars it can span multiple rows/columns. Confusing non-star
//   regions of the image (such as the lunar terminator) can also yield
//   multi-pixel cores.
// Neighbors: the single pixel box surrounding the core.
// Margin: the single pixel box surrounding neighbors.
// Perimeter: the single pixel box surrounding margin.
//
// When the core contains a star, we will observe/require the following:
//
// * Core is not too large.
// * Core is brightest.
// * The perimeter pixels are taken to reflect the local sky background value.
// * The core is brighter than the sky background by a statistically significant
//   amount.
// * The neighbor box might not be as bright as the core, but it will brighter
//   than some fraction of the core. This criterion lets us reject single hot
//   pixels.
//
// If a Blob and its surroundings satisfies these (and other; see the code)
// criteria, a StarDescription is generated by performing centroiding on the
// core+neighbor pixels. If any of these criteria are not met, the Blob
// is deemed to be NOT a star.
//
// Note that the "margin" box isn't mentioned in the above narration. The margin
// box allows some additional star defocus by pushing the sky background pixels
// (perimeter box) out a bit.
//
// Statistical significance is defined as a `sigma` multiple of the
// `noise_estimate`.
fn gate_star_2d(blob: &Blob, image: &GrayImage,
                noise_estimate: f32, sigma: f32,
                max_width: u32, max_height: u32) -> Option<StarDescription> {
    let (image_width, image_height) = image.dimensions();
    // Compute the bounding box of all of the blob's center coords.
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
    let core_x_min = x_min as i32;
    let core_x_max = x_max as i32;
    let core_y_min = y_min as i32;
    let core_y_max = y_max as i32;
    let core_width = (core_x_max - core_x_min) as u32 + 1;
    let core_height = (core_y_max - core_y_min) as u32 + 1;

    // Define bounding box for core.
    let core = Rect::at(core_x_min, core_y_min)
        .of_size(core_width, core_height);

    // Reject blob if it is too big.
    if core_width > max_width || core_height > max_height {
        debug!("Blob {:?} too large", core);
        return None;
    }
    // Reject blob if its expansion goes past an image boundary.
    if core_x_min - 3 < 0 || core_x_max + 3 >= image_width as i32 ||
        core_y_min - 3 < 0 || core_y_max + 3 >= image_height as i32
    {
        debug!("Blob {:?} too close to edge", core);
        return None;
    }

    // Expand core bounding box by three pixels in all directions.
    let neighbors = Rect::at(core_x_min - 1, core_y_min - 1)
        .of_size(core_width + 2, core_height + 2);
    let margin = Rect::at(core_x_min - 2, core_y_min - 2)
        .of_size(core_width + 4, core_height + 4);
    let perimeter = Rect::at(core_x_min - 3, core_y_min - 3)
        .of_size(core_width + 6, core_height + 6);

    // Compute average of pixels in core.
    let mut core_sum: i32 = 0;
    let mut core_count: i32 = 0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        &image, &core, /*include_interior=*/true) {
        core_sum += pixel_value as i32;
        core_count += 1;
    }
    let core_mean = core_sum as f32 / core_count as f32;

    if core_width >= 3 && core_height >= 3 {
        // Require that the "inner" core be at least as bright as the outer
        // core. This covers the case where the blob's candidates are e.g. a
        // sunlit crater rim at the lunar terminator. These candidates could
        // form an arc shape with a dark interior. Usually such cases will be
        // rejected by the limit on the blob size, but we add an inner core
        // brightness criterion here for completeness.
        let mut outer_core_sum: i32 = 0;
        let mut outer_core_count: i32 = 0;
        for (_x, _y, pixel_value) in EnumeratePixels::new(
            &image, &core, /*include_interior=*/false) {
            outer_core_sum += pixel_value as i32;
            outer_core_count += 1;
        }
        let outer_core_mean = outer_core_sum as f32 / outer_core_count as f32;
        // When including the inner core (core_mean), we should be at least as
        // bright as when excluding the inner core (outer_core_mean).
        if core_mean < outer_core_mean {
            debug!("Overall core average {} is less than outer core average {} for blob {:?}",
                   core_mean, outer_core_mean, core);
            return None;
        }
    }

    // Compute average of pixels in box immediately surrounding core.
    let mut neighbor_sum: i32 = 0;
    let mut neighbor_count: i32 = 0;
    for (x, y, pixel_value) in EnumeratePixels::new(
        &image, &neighbors, /*include_interior=*/false) {
        let is_corner =
            (x == neighbors.left() || x == neighbors.right()) &&
            (y == neighbors.top() || y == neighbors.bottom());
        if is_corner {
            continue;  // Exclude corner pixels.
        }
        neighbor_sum += pixel_value as i32;
        neighbor_count += 1;
    }
    let neighbor_mean = neighbor_sum as f32 / neighbor_count as f32;
    // Core average must be at least as bright as the neighbor average.
    if core_mean < neighbor_mean {
        debug!("Core average {} is less than neighbor average {} for blob {:?}",
               core_mean, neighbor_mean, core);
        return None;
    }

    // Compute average of pixels in next box out; this is one pixel
    // inward from the outer perimeter.
    let mut margin_sum: i32 = 0;
    let mut margin_count: i32 = 0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        &image, &margin, /*include_interior=*/false) {
        margin_sum += pixel_value as i32;
        margin_count += 1;
    }
    let margin_mean = margin_sum as f32 / margin_count as f32;
    if core_mean <= margin_mean {
        debug!("Core average {} is not greater than margin average {} for blob {:?}",
               core_mean, margin_mean, core);
        return None;
    }

    // Gather information from the outer perimeter. These pixels represent
    // the background.
    let mut perimeter_sum: i32 = 0;
    let mut perimeter_count: i32 = 0;
    let mut perimeter_min = u8::MAX;
    let mut perimeter_max = 0_u8;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        &image, &perimeter, /*include_interior=*/false) {
        perimeter_sum += pixel_value as i32;
        perimeter_count += 1;
        perimeter_min = cmp::min(perimeter_min, pixel_value);
        perimeter_max = cmp::max(perimeter_max, pixel_value);
    }
    let background_est = perimeter_sum as f32 / perimeter_count as f32;
    debug!("background: {} for blob {:?}", background_est, core);

    // We require the perimeter pixels to be ~uniformly dark. See if any
    // perimeter pixel is too bright compared to the darkest perimeter
    // pixel.
    // The 3/2 sigma_noise threshold is empirically chosen to yield a low
    // rejection rate for actual sky background perimeter pixels.
    if (perimeter_max - perimeter_min) as f32 > 1.5 * sigma * noise_estimate {
        debug!("Perimeter too varied for blob {:?}", core);
        return None;
    }

    // Verify that core average exceeds background by sigma * noise.
    if core_mean - background_est < sigma * noise_estimate {
        debug!("Core too weak for blob {:?}", core);
        return None;
    }
    if core_width == 1 && core_height == 1 {
        // Verify that the neighbor average (minus background) exceeds 0.25 *
        // core (minus background).
        if neighbor_mean - background_est <= 0.25 * (core_mean - background_est) {
            // Hot pixel.
            debug!("Neighbors too weak for blob {:?}", core);
            return None;
        }
    }
    // Star passes all of the 2d gates!
    Some(create_star_description(image, &neighbors, background_est))
}

// Called when gate_star_2d() determines that a Blob is detected as containing a
// star.
// neighbors: specifies the region encompassing the Blob plus a one pixel
//     surround.
// background_est: the average value of the "perimeter" pixels around the Blob.
fn create_star_description(image: &GrayImage, neighbors: &Rect, background_est: f32)
                           -> StarDescription {
    // Process the interior pixels (core plus immediate neighbors) to
    // obtain moments. Also note the count of saturated pixels.
    let mut num_saturated = 0;
    let mut m0: f32 = 0.0;
    let mut m1x: f32 = 0.0;
    let mut m1y: f32 = 0.0;
    for (x, y, pixel_value) in EnumeratePixels::new(
        &image, &neighbors, /*include_interior=*/true) {
        if pixel_value == 255 {
            num_saturated += 1;
        }
        let val_minus_bkg = pixel_value as f32 - background_est;
        m0 += val_minus_bkg;
        m1x += x as f32 * val_minus_bkg;
        m1y += y as f32 * val_minus_bkg;
    }
    // We use simple center-of-mass as the centroid.
    let centroid_x = m1x / m0;
    let centroid_y = m1y / m0;
    // Compute second moment about the centroid.
    let mut m2x_c: f32 = 0.0;
    let mut m2y_c: f32 = 0.0;
    for (x, y, pixel_value) in EnumeratePixels::new(
        &image, &neighbors, /*include_interior=*/true) {
        let val_minus_bkg = pixel_value as f32 - background_est;
        m2x_c += (x as f32 - centroid_x) * (x as f32 - centroid_x) * val_minus_bkg;
        m2y_c += (y as f32 - centroid_y) * (y as f32 - centroid_y) * val_minus_bkg;
    }
    let variance_x = m2x_c / m0;
    let variance_y = m2y_c / m0;
    StarDescription{centroid_x: (centroid_x + 0.5) as f32,
                    centroid_y: (centroid_y + 0.5) as f32,
                    stddev_x: variance_x.sqrt() as f32,
                    stddev_y: variance_y.sqrt() as f32,
                    mean_brightness:
                    m0 / (neighbors.width() * neighbors.height()) as f32,
                    background: background_est,
                    num_saturated}
}

/// Estimates the RMS noise of the given image. A small portion of the image
/// is processed as follows:
/// 1. The brightest pixels are excluded.
/// 2. The standard deviation of the remaining pixels is computed.
///
/// To guard against accidentally sampling a bright part of the image (moon?
/// streetlamp?), we sample a few image regions and choose the darkest one to
/// measure the noise.
pub fn estimate_noise_from_image(image: &GrayImage) -> f32 {
    let noise_start = Instant::now();
    let (width, height) = image.dimensions();
    let box_size = cmp::min(30, cmp::min(width, height) / 4);
    let mut stats_vec = Vec::<(f32, f32)>::new();
    // Sample three areas across the horizontal midline of the image.
    stats_vec.push(stats_for_roi(image, &Rect::at(
        (width*1/4 - box_size/2) as i32, (height/2 - box_size/2) as i32)
                                 .of_size(box_size, box_size)));
    stats_vec.push(stats_for_roi(image, &Rect::at(
        (width*2/4 - box_size/2) as i32, (height/2 - box_size/2) as i32)
                                 .of_size(box_size, box_size)));
    stats_vec.push(stats_for_roi(image, &Rect::at(
        (width*3/4 - box_size/2) as i32, (height/2 - box_size/2) as i32)
                                 .of_size(box_size, box_size)));
    // Pick the darkest box by median value.
    stats_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let stddev = stats_vec[0].1;
    debug!("Noise estimate {} found in {:?}", stddev, noise_start.elapsed());
    stddev
}

// Returns (median, stddev) for the given image region. Excludes bright pixels
// that are likely to be stars, as we're interested in the median/stddev of the
// sky background.
fn stats_for_roi(image: &GrayImage, roi: &Rect) -> (/*median*/f32, /*stddev*/f32) {
    let mut histogram: [u32; 256] = [0; 256];
    let mut pixel_count = 0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        &image, &roi, /*include_interior=*/true) {
        histogram[pixel_value as usize] += 1;
        pixel_count += 1;
    }
    // Do a sloppy trim of the brightest pixels; this will give us a de-starred
    // median and stddev that we can use for a more precise trim.
    let trimmed_histogram = trim_histogram(&histogram, pixel_count * 9 / 10);
    let (_trimmed_mean, trimmed_stddev, trimmed_median) =
        stats_for_histogram(&trimmed_histogram);
    debug!("Original histogram: {:?}; median {}; stddev {}",
           histogram, trimmed_median, trimmed_stddev);
    // Any pixel whose value is N * stddev above the median is deemed a star and
    // kicked out of the histogram.
    let star_cutoff = (trimmed_median as f32 +
                       8.0 as f32 * f32::max(trimmed_stddev, 1.0)) as usize;
    for h in 0_usize..256 {
        if h >= star_cutoff {
            histogram[h] = 0;
        }
    }
    debug!("De-starred histogram: {:?}", histogram);
    let (_mean, stddev, median) = stats_for_histogram(&histogram);
    (median as f32, stddev)
}

fn trim_histogram(histogram: &[u32; 256], count_to_keep: u32)
                  -> [u32; 256] {
    let mut trimmed_histogram = *histogram;
    let mut count = 0;
    for h in 0..256 {
        let bin_count = trimmed_histogram[h];
        if count + bin_count > count_to_keep {
            let excess = count + bin_count - count_to_keep;
            trimmed_histogram[h] -= excess;
        }
        count += trimmed_histogram[h];
    }
    return trimmed_histogram;
}

fn stats_for_histogram(histogram: &[u32; 256])
                       -> (/*mean*/f32, /*stddev*/f32, /*median*/usize) {
    let mut count = 0;
    let mut first_moment = 0;
    for h in 0..256 {
        let bin_count = histogram[h];
        count += bin_count;
        first_moment += bin_count * h as u32;
    }
    if count == 0 {
        return (0.0, 0.0, 0);
    }
    let mean = first_moment as f32 / count as f32;
    let mut second_moment: f32 = 0.0;
    let mut sub_count = 0;
    let mut median = 0;
    for h in 0..256 {
        let bin_count = histogram[h];
        second_moment += bin_count as f32 * (h as f32 - mean) * (h as f32 - mean);
        if sub_count < count / 2 {
            sub_count += bin_count;
            if sub_count >= count / 2 {
                median = h;
            }
        }
    }
    let stddev = (second_moment / count as f32).sqrt();
    (mean, stddev, median)
}

/// This function runs the StarGate algorithm on the supplied `image`, returning
/// a [StarDescription] for each detected star.
///
/// # Arguments
///   `image` - The image to analyze. The entire image is scanned for stars,
///   excluding the three leftmost and three rightmost columns.
///
///   `noise_estimate` The noise level of `image`. This is typically the noise
///   level returned by [estimate_noise_from_image()].
///
///   `sigma` - Specifies the statistical significance threshold used for
///   discriminating stars from background. Given a noise measure N, a pixel's
///   value must be at least `sigma`*N greater than the background value in order
///   to be considered a star candidate. Higher `sigma` values yield fewer
///   stars; lower `sigma` values yield more stars but increase the likelihood
///   of noise-induced false positives. Typical `sigma` values: 5-10.
///
///   `max_size` - StarGate clumps adjacent bright pixels to form a single star
///   candidate. The `max_size` argument governs how large a clump can be before
///   it is rejected. Note that making `max_size` small can eliminate very
///   bright stars that "bleed" to many surrounding pixels. Typical `max_size`
///   values: 3-5.
///
/// # Returns
/// Vec<[StarDescription]>, in unspecified order.
///
/// u32: The number of hot pixels seen. See implementation for more information
/// about hot pixels.
pub fn get_stars_from_image(image: &GrayImage,
                            noise_estimate: f32, sigma: f32, max_size: u32)
                            -> (Vec<StarDescription>, /*hot_pixel_count*/u32)
{
    // If noise estimate is below 0.5, assume that the image background has been
    // crushed to black; use a minimum noise value.
    let corrected_noise_estimate = f32::max(noise_estimate, 0.5);

    let mut stars = Vec::<StarDescription>::new();
    let (candidates, hot_pixel_count) =
        scan_image_for_candidates(image, corrected_noise_estimate, sigma);
    for blob in form_blobs_from_candidates(candidates) {
        match gate_star_2d(&blob, image, corrected_noise_estimate,
                           sigma, max_size, max_size) {
            Some(x) => stars.push(x),
            None => ()
        }
    }
    // A hot pixel is defined to be a single bright pixel whose immediate
    // neighbors are at sky background level. Normally the hot pixel count is
    // ~constant for a given image detector. get_stars_from_image() returns this
    // value to allow application logic to detect situations where too-sharp
    // focus with large pixels can cause stars to be mis-classified as hot
    // pixels. From an initial defocused state, as focus is improved, the number
    // of detected stars rises, but as we advance into the too-focused regime,
    // the number of detected star candidates will drop as the number of
    // reported hot pixels rises. A rising hot pixel count can be a cue to the
    // application logic that over-focusing is happening.
    (stars, hot_pixel_count)
}

/// Summarizes an image region of interest. The pixel values used are not
/// background subtracted. Single hot pixels are replaced with interpolated
/// neighbor values when locating the peak pixel and when accumulating the
/// histogram.
#[derive(Debug)]
#[allow(dead_code)]
pub struct RegionOfInterestSummary {
    /// Histogram of pixel values in the ROI.
    pub histogram: [u32; 256],

    /// Each element is the mean of a row of the ROI. Size is thus the ROI height.
    pub horizontal_projection: Vec<f32>,  // TODO: drop this

    /// Each element is the mean of a column of the ROI. Size is thus the ROI
    /// width.
    pub vertical_projection: Vec<f32>,  // TODO: drop this

    /// The location (in image coordinates) of the peak pixel (after correcting
    /// for hot pixels). If there are multiple pixels with the peak value, it is
    /// unspecified which one's location is reported here. The application logic
    /// should use `histogram` to adjust exposure to avoid too many peak
    /// (saturated) pixels.
    pub peak_x: i32,
    pub peak_y: i32,
}

/// Gathers information from a region of interest of an image.
///
/// # Arguments
///   `image` - The image of which a portion will be summarized.
///
///   `roi` - Specifies the portion of `image` to be summarized.
///
///   `noise_estimate` The noise level of `image`. This is typically the noise
///   level returned by [estimate_noise_from_image()].
///
///   `sigma` - Specifies the statistical significance threshold used for
///   discriminating hot pixels from background. This can be the same value as
///   passed to [get_stars_from_image()].
///
/// # Returns
/// [RegionOfInterestSummary] Information that compactly provides some information
/// about the ROI. The `histogram` result can be used by an application's
/// auto-exposure logic; the `horizontal_projection` and `vertical_projection`
/// results can be used by an application to locate the brightest star in the
/// ROI, even if it is severely out of focus.
///
/// # Panics
/// The `roi` must exclude the three leftmost and three rightmost image columns.
pub fn summarize_region_of_interest(image: &GrayImage, roi: &Rect,
                                    noise_estimate: f32, sigma: f32)
                                    -> RegionOfInterestSummary {
    let process_roi_start = Instant::now();

    let mut peak_x = 0;
    let mut peak_y = 0;
    let mut peak_val = 0_u8;
    let (width, height) = image.dimensions();
    assert!(roi.bottom() < height as i32);
    // Sliding gate needs to extend past left and right edges of ROI. Make sure
    // there's enough image.
    let gate_leftmost: i32 = roi.left() as i32 - 3;
    let gate_rightmost = roi.right() + 4;  // One past.
    assert!(gate_leftmost >= 0);
    assert!(gate_rightmost <= width as i32);
    let image_pixels: &Vec<u8> = image.as_raw();

    let mut histogram: [u32; 256] = [0_u32; 256];
    let mut horizontal_projection_sum = Vec::<u32>::new();
    let mut vertical_projection_sum = Vec::<u32>::new();
    horizontal_projection_sum.resize(roi.height() as usize, 0_u32);
    vertical_projection_sum.resize(roi.width() as usize, 0_u32);

    let sigma_noise_2 = cmp::max((2.0 * sigma * noise_estimate) as i16, 2);
    let sigma_noise_1_5 = cmp::max((1.5 * sigma * noise_estimate) as i16, 1);
    for rownum in roi.top()..=roi.bottom() {
        // Get the slice of image_pixels corresponding to this row of the ROI.
        let row_start = (rownum * width as i32) as usize;
        let row_pixels: &[u8] = &image_pixels.as_slice()
            [row_start + gate_leftmost as usize ..
             row_start + gate_rightmost as usize];
        // Slide a 7 pixel gate across the row.
        let mut center_x = roi.left();
        for gate in row_pixels.windows(7) {
            let (pixel_value, _result_type) =
                gate_star_1d(gate, sigma_noise_2, sigma_noise_1_5);
            histogram[pixel_value as usize] += 1;
            horizontal_projection_sum[(rownum - roi.top()) as usize]
                += pixel_value as u32;
            vertical_projection_sum[(center_x - roi.left()) as usize]
                += pixel_value as u32;
            if pixel_value > peak_val {
                peak_x = center_x;
                peak_y = rownum;
                peak_val = pixel_value;
            }
            center_x += 1;
        }
    }
    let h_proj: Vec<f32> = horizontal_projection_sum.into_iter().map(
        |x| x as f32 / roi.width() as f32).collect();
    let v_proj: Vec<f32> = vertical_projection_sum.into_iter().map(
        |x| x as f32 / roi.height() as f32).collect();

    debug!("ROI processing completed in {:?}", process_roi_start.elapsed());
    RegionOfInterestSummary{histogram,
                            horizontal_projection: h_proj,
                            vertical_projection: v_proj,
                            peak_x, peak_y,
    }
}

#[cfg(test)]
mod tests {
    extern crate approx;
    use approx::assert_abs_diff_eq;
    use image::Luma;
    use imageproc::gray_image;
    use imageproc::noise::gaussian_noise;
    use super::*;

    #[test]
    #[should_panic]
    fn test_enumerate_pixels_roi_too_large() {
        let empty_image = gray_image!();
        let _iter = EnumeratePixels::new(
            &empty_image,
            &Rect::at(0, 0).of_size(1, 1),
            /*include_interior*/false);
    }

    #[test]
    fn test_enumerate_pixels_1x1() {
        let image_1x1 = gray_image!(127);
        let pixels: Vec<(i32, i32, u8)> =
            EnumeratePixels::new(&image_1x1,
                                 &Rect::at(0, 0).of_size(1, 1),
                                 /*include_interior*/false).collect();
        assert_eq!(pixels, vec!((0, 0, 127)));
    }

    #[test]
    fn test_enumerate_pixels_3x3() {
        let image_3x3 = gray_image!(
            0, 1, 2;
            127, 253, 254;
            255, 0, 1);

        // Entire ROI, with interior.
        let mut pixels: Vec<(i32, i32, u8)> =
            EnumeratePixels::new(&image_3x3,
                                 &Rect::at(0, 0).of_size(3, 3),
                                 /*include_interior*/true).collect();
        assert_eq!(pixels, vec!((0, 0, 0),
                                (1, 0, 1),
                                (2, 0, 2),
                                (0, 1, 127),
                                (1, 1, 253),
                                (2, 1, 254),
                                (0, 2, 255),
                                (1, 2, 0),
                                (2, 2, 1)));
        // Entire ROI, no interior.
        pixels = EnumeratePixels::new(&image_3x3,
                                      &Rect::at(0, 0).of_size(3, 3),
                                      /*include_interior*/false).collect();
        assert_eq!(pixels, vec!((0, 0, 0),
                                (1, 0, 1),
                                (2, 0, 2),
                                (0, 1, 127),
                                (2, 1, 254),
                                (0, 2, 255),
                                (1, 2, 0),
                                (2, 2, 1)));
    }

    #[test]
    fn test_stats_for_roi() {
        let mut image_5x4 = GrayImage::new(5, 4);
        // Leave image as all zeros for now.
        let (mut median, mut stddev) = stats_for_roi(
            &image_5x4,
            &Rect::at(0, 0).of_size(5, 4));
        assert_eq!(median, 0_f32);
        assert_eq!(stddev, 0_f32);

        // Add some noise.
        image_5x4.put_pixel(1, 0, Luma::<u8>([1]));
        image_5x4.put_pixel(2, 0, Luma::<u8>([2]));
        image_5x4.put_pixel(3, 0, Luma::<u8>([1]));
        image_5x4.put_pixel(4, 0, Luma::<u8>([2]));
        image_5x4.put_pixel(0, 1, Luma::<u8>([1]));
        (median, stddev) = stats_for_roi(
            &image_5x4,
            &Rect::at(0, 0).of_size(5, 4));
        assert_eq!(median, 0_f32);
        assert_abs_diff_eq!(stddev, 0.65, epsilon = 0.01);

        // Single bright pixel. This is removed because we eliminate the
        // brightest pixel(s).
        image_5x4.put_pixel(0, 0, Luma::<u8>([255]));
        (median, stddev) = stats_for_roi(
            &image_5x4,
            &Rect::at(0, 0).of_size(5, 4));
        assert_eq!(median, 0_f32);
        assert_abs_diff_eq!(stddev, 0.66, epsilon = 0.01);

        // Add another non-zero pixel. This will be kept.
        image_5x4.put_pixel(0, 1, Luma::<u8>([4]));
        (median, stddev) = stats_for_roi(
            &image_5x4,
            &Rect::at(0, 0).of_size(5, 4));
        assert_eq!(median, 0_f32);
        assert_abs_diff_eq!(stddev, 1.04, epsilon = 0.01);
    }

    #[test]
    fn test_stats_for_histogram() {
        let mut histogram = [0_u32; 256];
        histogram[10] = 2;
        histogram[20] = 2;
        let (mean, stddev, median) = stats_for_histogram(&histogram);
        assert_eq!(mean, 15.0);
        assert_eq!(stddev, 5.0);
        assert_eq!(median, 10);
    }

    #[test]
    fn test_estimate_noise_from_image() {
        let small_image = gaussian_noise(&GrayImage::new(100, 100),
                                         10.0, 3.0, 42);
        // The stddev is what we requested because there are no bright
        // outliers to discard.
        assert_abs_diff_eq!(estimate_noise_from_image(&small_image),
                            3.0, epsilon = 0.1);
        let large_image = gaussian_noise(&GrayImage::new(1000, 1000),
                                         10.0, 3.0, 42);
        assert_abs_diff_eq!(estimate_noise_from_image(&large_image),
                            3.0, epsilon = 0.1);
    }

    #[test]
    fn test_gate_star_1d_center_bright_wrt_background() {
        // Center minus background not bright enough.
        let mut gate: [u8; 7] = [10, 10, 10, 12, 10, 10, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 12);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Center minus background is bright enough.
        gate = [10, 10, 11, 13, 11, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_center_bright_wrt_neighbor() {
        // Center is less than a neighbor.
        let mut gate: [u8; 7] = [10, 10, 11, 13, 14, 10, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Ditto, other neighbor.
        gate = [10, 10, 14, 13, 11, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Center is at least as bright as its neighbors. Tie break is to
        // left (current candidate); this is explored further below.
        gate = [10, 10, 11, 13, 13, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_center_bright_wrt_margin() {
        // Center is not brighter than a margin.
        let mut gate: [u8; 7] = [10, 10, 11, 13, 11, 13, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Ditto, other margin.
        gate = [10, 13, 11, 13, 11, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Center brighter than both margins.
        gate = [10, 12, 11, 13, 11, 12, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_left_center_tie() {
        // When center and left have same value, tie is broken
        // based on next surrounding values.
        // In this case, the tie breaks to the left.
        let mut gate: [u8; 7] = [10, 11, 13, 13, 10, 11, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Here, the tie breaks to the right (center pixel).
        gate = [10, 11, 13, 13, 11, 11, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_center_right_tie() {
        // When center and right have same value, tie is broken
        // based on next surrounding values.
        // In this case, the tie breaks to the right.
        let mut gate: [u8; 7] = [10, 11, 11, 13, 13, 11, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Here, the tie breaks to the left (center pixel).
        gate = [10, 11, 11, 13, 13, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_hot_pixel() {
        // Neighbors are too dark, so bright center is deemed a hot pixel.
        let mut gate: [u8; 7] = [10, 10, 10, 15, 12, 10, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4);
        assert_eq!(value, 11);
        assert_eq!(result_type, ResultType::HotPixel);

        // Neighbors have enough brighness, so bright center is deemed a
        // star candidate.
        gate = [10, 10, 12, 15, 12, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/2);
        assert_eq!(value, 15);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_unequal_border() {
        // Border pixels differ too much, so candidate is rejected.
        let mut gate: [u8; 7] = [12, 10, 12, 18, 13, 10, 7];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4);
        assert_eq!(value, 18);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Borders are close enough now.
        gate = [11, 10, 12, 18, 13, 10, 7];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4);
        assert_eq!(value, 18);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_form_blobs_from_candidates() {
        let mut candidates = Vec::<CandidateFrom1D>::new();
        // Candidates on the same row are not combined, even if they are close
        // together. This is because, in practice due to the operation of
        // gate_star_1d(), candiates in the same row will always be well
        // separated.
        candidates.push(CandidateFrom1D{x: 20, y:3});
        candidates.push(CandidateFrom1D{x: 23, y:3});
        let mut blobs = form_blobs_from_candidates(candidates);
        blobs.sort_by(|a, b| a.candidates[0].x.cmp(&b.candidates[0].x));
        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[0].candidates.len(), 1);
        assert_eq!(blobs[0].candidates[0].x, 20);
        assert_eq!(blobs[0].candidates[0].y, 3);
        assert_eq!(blobs[1].candidates.len(), 1);
        assert_eq!(blobs[1].candidates[0].x, 23);
        assert_eq!(blobs[1].candidates[0].y, 3);

        // Candidates on adjacent rows are combined if they are close
        // enough w.r.t. their horizontal offset.
        // Not combined:
        // . . . . . A . . . . . .
        // . B . . . . . . . . . .
        candidates = Vec::<CandidateFrom1D>::new();
        candidates.push(CandidateFrom1D{x: 5, y:0});  // A.
        candidates.push(CandidateFrom1D{x: 1, y:1});  // B.
        blobs = form_blobs_from_candidates(candidates);
        assert_eq!(blobs.len(), 2);

        // Combined:
        // . . . . A . . . . . . .
        // . B . . . . . . . . . .
        candidates = Vec::<CandidateFrom1D>::new();
        candidates.push(CandidateFrom1D{x: 4, y:0});  // A.
        candidates.push(CandidateFrom1D{x: 1, y:1});  // B.
        blobs = form_blobs_from_candidates(candidates);
        assert_eq!(blobs.len(), 1);

        // Combined:
        // . . . . A . . . . . . .
        // . . . . B . . . . . . .
        candidates = Vec::<CandidateFrom1D>::new();
        candidates.push(CandidateFrom1D{x: 4, y:0});  // A.
        candidates.push(CandidateFrom1D{x: 4, y:1});  // B.
        blobs = form_blobs_from_candidates(candidates);
        assert_eq!(blobs.len(), 1);

        // Combined:
        // . . . . A . . . . . . .
        // . . . . . . . B . . . .
        candidates = Vec::<CandidateFrom1D>::new();
        candidates.push(CandidateFrom1D{x: 4, y:0});  // A.
        candidates.push(CandidateFrom1D{x: 7, y:1});  // B.
        blobs = form_blobs_from_candidates(candidates);
        assert_eq!(blobs.len(), 1);

        // Not combined:
        // . . . . A . . . . . . .
        // . . . . . . . . B . . .
        candidates = Vec::<CandidateFrom1D>::new();
        candidates.push(CandidateFrom1D{x: 4, y:0});  // A.
        candidates.push(CandidateFrom1D{x: 8, y:1});  // B.
        blobs = form_blobs_from_candidates(candidates);
        assert_eq!(blobs.len(), 2);

        // In this case, B absorbs A and C. When D is reached, it sees that C
        // has been absorbed into B and D thus absorbs B (and its already
        // absorbed A and C).
        // . A . . . C . . . . . .
        // . . . B . . . D . . . .
        candidates = Vec::<CandidateFrom1D>::new();
        candidates.push(CandidateFrom1D{x: 1, y:0});  // A.
        candidates.push(CandidateFrom1D{x: 3, y:1});  // B.
        candidates.push(CandidateFrom1D{x: 5, y:0});  // C.
        candidates.push(CandidateFrom1D{x: 7, y:1});  // D.
        blobs = form_blobs_from_candidates(candidates);
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].candidates.len(), 4);
    }

    #[test]
    fn test_gate_star_2d_too_large() {
        let mut blob = Blob{candidates: Vec::<CandidateFrom1D>::new(),
                            recipient_blob: -1};
        let image_9x9 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9, 11, 11, 11, 11, 11,  9,  9;
            9,  9, 11, 18, 20, 19, 11,  9,  9;
            9,  9, 11, 20, 30, 20, 11,  9,  9;
            9,  9, 11, 19, 20, 19, 11,  9,  9;
            9,  9, 11, 11, 11, 11, 11,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9);
        // Make 3x3 blob.
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        blob.candidates.push(CandidateFrom1D{x: 5, y: 5});
        // 3x3 is too big.
        match gate_star_2d(&blob, &image_9x9, 1.0, 6.0,
                           /*max_width=*/3, /*max_height=*/2) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        match gate_star_2d(&blob, &image_9x9, 1.0, 6.0,
                           /*max_width=*/2, /*max_height=*/3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // We allow a 3x3 blob here.
        match gate_star_2d(&blob, &image_9x9, 1.0, 6.0,
                           /*max_width=*/3, /*max_height=*/3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate")
        }
    }

    #[test]
    fn test_gate_star_2d_image_boundary() {
        let mut blob = Blob{candidates: Vec::<CandidateFrom1D>::new(),
                            recipient_blob: -1};
        let image_7x7 = gray_image!(
        9,  9,  9,  9,  9,  9,  9;
        9, 10, 10, 10, 10, 10,  9;
        9, 10, 12, 15, 12, 10,  9;
        9, 10, 15, 30, 15, 10,  9;
        9, 10, 12, 15, 12, 10,  9;
        9, 10, 10, 10, 10, 10,  9;
        9,  9,  9,  9,  9,  9,  9);
        // Make 1x1 blob, but too far to the left.
        blob.candidates.push(CandidateFrom1D{x: 2, y: 3});
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Too far to right.
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 4, y: 3});
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Too high.
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 3, y: 2});
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Too low.
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 3, y: 4});
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Just right!
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate"),
        }
    }

    #[test]
    fn test_gate_star_2d_hollow_core() {
        let mut blob = Blob{candidates: Vec::<CandidateFrom1D>::new(),
                            recipient_blob: -1};
        let mut image_9x9 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9, 11, 11, 11, 11, 11,  9,  9;
            9,  9, 11, 19, 20, 19, 11,  9,  9;
            9,  9, 11, 20, 19, 20, 11,  9,  9;
            9,  9, 11, 19, 20, 19, 11,  9,  9;
            9,  9, 11, 11, 11, 11, 11,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9);
        // Make 3x3 blob.
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        blob.candidates.push(CandidateFrom1D{x: 5, y: 5});
        // Center of core is less bright than outer core.
        match gate_star_2d(&blob, &image_9x9, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_9x9.put_pixel(4, 4, Luma::<u8>([20]));
        match gate_star_2d(&blob, &image_9x9, 1.0, 6.0, 3, 3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate"),
        }
    }

    #[test]
    fn test_gate_star_2d_core_bright_wrt_neighbor() {
        let mut blob = Blob{candidates: Vec::<CandidateFrom1D>::new(),
                            recipient_blob: -1};
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        let mut image_7x7 = gray_image!(
        8,  8,  8,  8,  8,  8,  8;
        8, 10, 10, 10, 10, 10,  8;
        8, 10, 12, 14, 12, 10,  8;
        8, 10, 14, 13, 14, 10,  8;
        8, 10, 12, 14, 12, 10,  8;
        8, 10, 10, 10, 10, 10,  8;
        8,  8,  8,  8,  8,  8,  8);
        // 1x1 core is less bright than neighbor average. Note that the
        // neighbor corners are excluded.
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_7x7.put_pixel(3, 3, Luma::<u8>([14]));
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate"),
        }
    }

    #[test]
    fn test_gate_star_2d_core_bright_wrt_margin() {
        let mut blob = Blob{candidates: Vec::<CandidateFrom1D>::new(),
                            recipient_blob: -1};
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        let mut image_7x7 = gray_image!(
        8,  8,  8,  8,  8,  8,  8;
        8, 14, 14, 14, 14, 14,  8;
        8, 14, 12, 14, 12, 14,  8;
        8, 14, 14, 14, 14, 14,  8;
        8, 14, 12, 14, 12, 14,  8;
        8, 14, 14, 14, 14, 14,  8;
        8,  8,  8,  8,  8,  8,  8);
        // 1x1 core is not brighter than margin average.
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_7x7.put_pixel(3, 3, Luma::<u8>([15]));
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate"),
        }
    }

    #[test]
    fn test_gate_star_2d_nonuniform_perimeter() {
        let mut blob = Blob{candidates: Vec::<CandidateFrom1D>::new(),
                            recipient_blob: -1};
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        let mut image_7x7 = gray_image!(
        9,  9,  9,  9,  9,  9,  9;
        9, 10, 10, 10, 10, 10,  9;
       20, 10, 12, 15, 12, 10,  9;
        9, 10, 15, 30, 15, 10,  9;
        9, 10, 12, 15, 12, 10,  9;
        9, 10, 10, 10, 10, 10,  9;
        9,  9,  9,  9,  9,  9,  9);
        // Perimeter has an anomalously bright pixel.
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Repair the perimeter.
        image_7x7.put_pixel(0, 2, Luma::<u8>([12]));
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate"),
        }
    }

    #[test]
    fn test_gate_star_2d_core_bright_wrt_perimeter() {
        let mut blob = Blob{candidates: Vec::<CandidateFrom1D>::new(),
                            recipient_blob: -1};
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        let mut image_7x7 = gray_image!(
        8,  8,  8,  8,  8,  8,  8;
        8,  9,  9,  9,  9,  9,  8;
        8,  9, 10, 11, 10,  9,  8;
        8,  9, 11, 13, 11,  9,  8;
        8,  9, 10, 11, 10,  9,  8;
        8,  9,  9,  9,  9,  9,  8;
        8,  8,  8,  8,  8,  8,  8);
        // 1x1 core is not brighter enough than perimeter average.
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_7x7.put_pixel(3, 3, Luma::<u8>([14]));
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate"),
        }
    }

    #[test]
    fn test_gate_star_2d_neighbor_bright_wrt_perimeter() {
        let mut blob = Blob{candidates: Vec::<CandidateFrom1D>::new(),
                            recipient_blob: -1};
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        let mut image_7x7 = gray_image!(
        8,  8,  8,  8,  8,  8,  8;
        8,  9,  9,  9,  9,  9,  8;
        8,  9,  9,  9,  9,  9,  8;
        8,  9,  9, 14,  9,  9,  8;
        8,  9,  9,  9,  9,  9,  8;
        8,  9,  9,  9,  9,  9,  8;
        8,  8,  8,  8,  8,  8,  8);
        // Neighbor ring is not brighter enough than perimeter average.
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make neighbors bright enough.
        image_7x7.put_pixel(2, 2, Luma::<u8>([12]));
        image_7x7.put_pixel(3, 2, Luma::<u8>([12]));
        match gate_star_2d(&blob, &image_7x7, 1.0, 6.0, 3, 3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate"),
        }
    }

    #[test]
    fn test_create_star_description() {
        let image_7x7 = gray_image!(
        9,  9,  9,   9,  9,  9,  9;
        9, 10, 10,  10, 10, 10,  9;
        9, 10, 12, 255, 12, 10,  9;
        9, 10, 14, 255, 14, 10,  9;
        9, 10, 12,  14, 30, 10,  9;
        9, 10, 10,  10, 10, 10,  9;
        9,  9,  9,   9,  9,  9,  9);
        let neighbors = Rect::at(2, 2).of_size(3, 3);
        let star_description = create_star_description(&image_7x7, &neighbors,
                                                       /*background_est=*/9.01);
        assert_abs_diff_eq!(star_description.centroid_x,
                            3.53, epsilon = 0.01);
        assert_abs_diff_eq!(star_description.centroid_y,
                            3.08, epsilon = 0.01);
        assert_abs_diff_eq!(star_description.stddev_x,
                            0.27, epsilon = 0.01);
        assert_abs_diff_eq!(star_description.stddev_y,
                            0.59, epsilon = 0.01);
        assert_abs_diff_eq!(star_description.mean_brightness,
                            59.6, epsilon = 0.1);
        assert_abs_diff_eq!(star_description.background,
                            9.01, epsilon = 0.01);
        assert_eq!(star_description.num_saturated, 2);
    }

    #[test]
    #[should_panic]
    fn test_summarize_region_of_interest_left_edge() {
        let roi = Rect::at(2, 0).of_size(3, 2);
        let image_9x9 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9);
        // Cannot give ROI too close to left edges.
        let _roi_summary = summarize_region_of_interest(
            &image_9x9, &roi, 1.0, 5.0);
    }

    #[test]
    #[should_panic]
    fn test_summarize_region_of_interest_right_edge() {
        let roi = Rect::at(4, 0).of_size(3, 2);
        let image_9x9 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9);
        // Cannot give ROI too close to right edges.
        let _roi_summary = summarize_region_of_interest(
            &image_9x9, &roi, 1.0, 5.0);
    }

    #[test]
    fn test_summarize_region_of_interest_good_edges() {
        let roi = Rect::at(3, 0).of_size(3, 2);
        // 80 is a hot pixel, is replaced by interpolation of its left
        // and right neighbors.
        let image_9x9 = gray_image!(
            9,  9,  9,  7,  80,  9, 9,  9,  9;
            9,  9,  9,  11, 20, 32, 10,  9,  9);
        // ROI is correct distance from left+right edges.
        let roi_summary = summarize_region_of_interest(
            &image_9x9, &roi, 1.0, 5.0);
        assert_eq!(roi_summary.histogram[7], 1);
        assert_eq!(roi_summary.histogram[8], 1);
        assert_eq!(roi_summary.histogram[9], 1);
        assert_eq!(roi_summary.histogram[11], 1);
        assert_eq!(roi_summary.histogram[20], 1);
        assert_eq!(roi_summary.histogram[32], 1);
        assert_eq!(roi_summary.horizontal_projection, [8.0, 21.0]);
        assert_eq!(roi_summary.vertical_projection, [9.0, 14.0, 20.5]);
        // The hot pixel is not the peak.
        assert_eq!(roi_summary.peak_x, 5);
        assert_eq!(roi_summary.peak_y, 1);
    }

}  // mod tests.
