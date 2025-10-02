// Copyright (c) 2025 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

//! CedarDetect provides efficient and accurate detection of stars in sky images.
//! Given an image, CedarDetect returns a list of detected star centroids expressed
//! in image pixel coordinates.
//!
//! Features:
//!
//! * Employs localized thresholding to tolerate changes in background levels
//!   across the image.
//! * Adapts to different image exposure levels.
//! * Estimates noise in the image and adapts the star detection threshold
//!   accordingly.
//! * Automatically classifies and rejects isolated hot pixels.
//! * Rejects trailed objects such as aircraft lights or satellites.
//! * Tolerates the presence of bright interlopers such as the moon or
//!   streetlights.
//! * Simple function call interface with few parameters aside from the input
//!   image.
//! * Fast! On a Raspberry Pi 4B, the execution time per 1M image pixels is
//!   usually less than 10ms, even when several dozen stars are present in the
//!   image.
//!
//! # Intended applications
//!
//! CedarDetect is designed to be used with astrometry and plate solving systems
//! such as [Cedar-Solve](https://github.com/smroid/cedar-solve). It can also be
//! incorporated into satellite star trackers.
//!
//! A goal of CedarDetect is to allow such applications to achieve fast response
//! times. CedarDetect contributes to this by running quickly and by tolerating a
//! degree of image noise allowing for shorter imaging integration times.
//!
//! # Star detection fidelity
//!
//! Like any detection algorithm, CedarDetect produces both false negatives (failures
//! to detect actual stars) and false positives (spurious star detections). The
//! Caveats section below mentions some causes of false negatives.
//!
//! False positives can occur as you reduce the `sigma` parameter when calling
//! [get_stars_from_image()] in an attempt to increase the number of detected
//! stars. Although CedarDetect requires evidence from multiple pixels to qualify
//! each star detection (reducing the occurrence of false positives), noise
//! inevitably wins if you push `sigma` too low.
//!
//! # Algorithm
//!
//! CedarDetect spends most of its time on a single efficient pass over the input
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
//! CedarDetect is designed to identify only star-like spots. It is not a
//! generalized astronomical source extraction system.
//!
//! ## Crowding
//!
//! The criteria used by CedarDetect to efficiently detect stars are designed
//! around the characteristics of a star image's pixels compared to surrounding
//! pixels. CedarDetect can thus be confused when stars are too closely spaced,
//! or a star is close to a hot pixel, or if image shake causes star images to
//! be "doubled". Such situations will usually cause closely spaced stars to
//! fail to be detected. Note that for applications such as plate solving, this
//! is probably for the better.
//!
//! ## Imaging requirements
//!
//! * CedarDetect supports only 8-bit grayscale images; color images or images with
//!   greater bit depth must be converted before calling
//!   [get_stars_from_image()].
//! * The imaging exposure time and sensor gain are usually chosen by the caller
//!   to yield a desired number of faint star detections. In so doing, if a
//!   bright star is overexposed to a degree that it bleeds into too many
//!   adjacent pixels, CedarDetect will reject the bright star.
//! * Pixel scale and focusing are crucial:
//!   * If star images are too extended w.r.t. the pixel grid, CedarDetect might not
//!     detect the stars. You'll either need to tighten the focus or use the pixel
//!     binning option when calling [get_stars_from_image()].
//!   * If stars are too small w.r.t. the pixel grid, CedarDetect might
//!     mis-classify stars as isolated hot pixels. A simple remedy is to
//!     slightly defocus causing stars to occupy a central peak pixel with a bit
//!     of spread into immediately adjacent pixels.
//! * CedarDetect does not tolerate more than maybe a single pixel of motion blur.
//!
//! ## Centroid estimation
//!
//! CedarDetect's computes a sub-pixel centroid position for each detected star
//! using quadratic interpolation of the peak and its neighbors. This suffices
//! for many purposes, but a more exacting application might want to do its own
//! centroiding:
//!
//! * If a higher bit-depth image was converted to 8 bit for calling
//!   [get_stars_from_image()], centroiding in the original image may yield
//!   better accuracy.
//! * The application can use a more artful centroiding technique such as
//!   gaussian fitting.
//!
//! ## Lens distortions
//!
//! Optical imaging systems can produce distortions that affect CedarDetect in two
//! ways:
//!
//! * Optical aberrations can cause star images to deviate from ideal small
//!   condensed spots. CedarDetect will fail to detect stars that are too distorted,
//!   such as near the corners of a wide field image.
//! * Lens pincushion and/or barrel distortions can cause reported star positions
//!   to deviate from their true positions. CedarDetect reports star centroids in
//!   the image's x/y coordinates; it is up to the application to apply any needed
//!   lens distortion corrections when doing astrometry (e.g. plate solving).

use std::cmp;
use std::collections::hash_map::HashMap;
use std::time::Instant;

use image::GrayImage;
use imageproc::rect::Rect;
use log::{debug};

use crate::histogram_funcs::{HistogramStats,
                             remove_stars_from_histogram,
                             stats_for_histogram};
use crate::image_funcs::bin_and_histogram_2x2;

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
                assert_eq!(self.cur_x, self.roi.left());
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
// and the `sigma_noise_3` argument is 3x the sigma*noise value.
//
// Returns: Whether the gate's center pixel is a star candidate or is
//   uninteresting.
#[derive(Debug, Eq, PartialEq)]
enum ResultType {
    Uninteresting,
    Candidate,
}

fn gate_star_1d(gate: &[u8], sigma_noise_2: i16, sigma_noise_3: i16)
                -> ResultType {
    debug_assert!(sigma_noise_2 > 0);
    debug_assert!(sigma_noise_3 > 0);
    debug_assert!(sigma_noise_2 <= sigma_noise_3);
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

    // Center pixel must be sigma * estimated noise brighter than the estimated
    // background. Do this test first, because it eliminates the vast majority
    // of candidates.
    let est_background_2 = lb + rb;
    let center_minus_background_2 = c + c - est_background_2;
    if center_minus_background_2 < sigma_noise_2 {
        return ResultType::Uninteresting;
    }
    // Center pixel must be at least as bright as its immediate left/right
    // neighbors.
    if l > c || c < r {
        return ResultType::Uninteresting;
    }
    // Center pixel must be strictly brighter than its left/right margin.
    if lm >= c || c <= rm {
        return ResultType::Uninteresting;
    }
    if l == c {
        // Break tie between left and center.
        if lm > r {
            // Left will have been the center of its own candidate entry.
            return ResultType::Uninteresting;
        }
    }
    if c == r {
        // Break tie between center and right.
        if l <= rm {
            // Right will be the center of its own candidate entry.
            return ResultType::Uninteresting;
        }
    }
    // We require the border pixels to be ~uniformly dark. See if there is too
    // much brightness difference between the border pixels.
    // The 3x sigma_noise threshold is empirically chosen to yield a low
    // rejection rate for actual sky background border pixels.
    let border_diff = (lb - rb).abs();
    if border_diff > sigma_noise_3 {
        return ResultType::Uninteresting;
    }
    // We have a candidate star from our 1d analysis!
    debug!("candidate: {:?}", gate);
    ResultType::Candidate
}

#[derive(Copy, Clone, Debug)]
struct CandidateFrom1D {
    x: i32,
    y: i32,
}

#[derive(Debug, Eq, PartialEq)]
enum PixelHotType {
    Dark,  // Pixel is not bright compared to threshold.
    Bright,  // Pixel is bright compared to threshold but is not isolated (hot).
    Hot,  // Pixel is bright compared to threshold and is isolated (hot).
}

// Returns Dark if the center pixel is not bright, so is neither part of a
// star nor an isolated hot pixel.
// Returns Bright if the center pixel is bright enough to possibly be part
// of a star.
// Returns Hot if the center of the 7-pixel gate is an isolated hot pixel. In
// this case the pixel value is replaced with the average of its neighbors.
fn classify_pixel(gate: &[u8], sigma_noise_2: i16) -> (PixelHotType, u8)
{
    let lb = gate[0] as i16;
    let c = gate[3] as i16;
    let rb = gate[6] as i16;

    // Hot pixel must be bright compared to background.
    let est_background_2 = lb + rb;
    let center_minus_background_2 = c + c - est_background_2;
    if center_minus_background_2 < sigma_noise_2 {
        return (PixelHotType::Dark, gate[3]);
    }

    // Sum of l + r (minus background) must exceed 0.25 * center (minus
    // background). Otherwise, the center is a hot pixel.
    let l = gate[2] as i16;
    let r = gate[4] as i16;
    let neighbor_sum = l + r;
    let neighbor_sum_minus_background = neighbor_sum - est_background_2;
    if 4 * neighbor_sum_minus_background <= center_minus_background_2 / 2 {
        return (PixelHotType::Hot, (neighbor_sum / 2) as u8);
    }
    (PixelHotType::Bright, c as u8)
}

// Applies gate_star_1d() at all pixel positions of the image (excluding the few
// leftmost and rightmost columns) to identify star candidates for futher
// screening.
// `image` Image to be scanned.
// `noise_estimate` The noise level of the image. See estimate_noise_from_image().
// `sigma` Specifies the multiple of the noise level by which a pixel must exceed
//     the background to be considered a star candidate, in addition to
//     satisfying other criteria.
// `compute_histogram` Specifies whether histogram should be computed.
// Returns:
// Vec<CandidateFrom1D>: the identifed star candidates, in raster scan order.
// Option([u32; 256]): if requested, the histogram of the `image` pixel values.
//   Excludes the few leftmost and rightmost columns.
fn scan_image_for_candidates(image: &GrayImage,
                             noise_estimate: f64,
                             sigma: f64,
                             compute_histogram: bool)
                             -> (Vec<CandidateFrom1D>, Option<[u32; 256]>)
{
    let mut histogram = [0_u32; 256];

    let row_scan_start = Instant::now();
    let width = image.dimensions().0 as usize;
    let height = image.dimensions().1 as usize;
    let image_pixels: &Vec<u8> = image.as_raw();

    let sigma_noise_2 = cmp::max((2.0 * sigma * noise_estimate + 0.5) as i16, 2);
    let sigma_noise_3 = cmp::max((3.0 * sigma * noise_estimate + 0.5) as i16, 3);

    let estimated_candidates = (width * height) / 10000;
    let mut candidates = Vec::<CandidateFrom1D>::with_capacity(estimated_candidates);
    for rownum in 0..height {
        // Get the slice of image_pixels corresponding to this row.
        let row_pixels: &[u8] = &image_pixels.as_slice()
            [rownum * width .. (rownum+1) * width];

        // First pass: sample every cache line (64 bytes) to estimate row minimum.
        let mut row_min = 255u8;
        for i in (0..row_pixels.len()).step_by(64) {
            row_min = row_min.min(row_pixels[i]);
        }
        let threshold = row_min.saturating_add(sigma_noise_2 as u8 / 2);

        // Second pass: pixel loop, only create gates when needed.
        if compute_histogram {
            for center_x in 3..(row_pixels.len()-3) {
                let center_pixel = row_pixels[center_x];
                histogram[center_pixel as usize] += 1;
                if center_pixel >= threshold {
                    let gate = &row_pixels[center_x-3..center_x+4];
                    let result_type = gate_star_1d(
                        gate, sigma_noise_2, sigma_noise_3);
                    match result_type {
                        ResultType::Uninteresting => (),
                        ResultType::Candidate => {
                            candidates.push(CandidateFrom1D{x: center_x as i32,
                                                            y: rownum as i32});
                        },
                    }
                }
            }
        } else {
            // Identical except omits histogram update.
            for center_x in 3..(row_pixels.len()-3) {
                let center_pixel = row_pixels[center_x];
                if center_pixel >= threshold {
                    let gate = &row_pixels[center_x-3..center_x+4];
                    let result_type = gate_star_1d(
                        gate, sigma_noise_2, sigma_noise_3);
                    match result_type {
                        ResultType::Uninteresting => (),
                        ResultType::Candidate => {
                            candidates.push(CandidateFrom1D{x: center_x as i32,
                                                            y: rownum as i32});
                        },
                    }
                }
            }
        }
    }  // Iterate over rows.

    debug!("Image scan found {} candidates in {:?}",
           candidates.len(), row_scan_start.elapsed());
    (candidates, if compute_histogram {Some(histogram)} else {None} )
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
    // Create an initial singular blob for each candidate.
    for (next_blob_id, candidate) in candidates.into_iter().enumerate() {
        blobs.insert(next_blob_id as i32, Blob{candidates: vec![candidate],
                                        recipient_blob: -1});
        if candidate.y as usize >= labeled_candidates_by_row.len() {
            labeled_candidates_by_row.resize(candidate.y as usize + 1,
                                             Vec::<LabeledCandidate>::new());
        }
        labeled_candidates_by_row[candidate.y as usize].push(
            LabeledCandidate{candidate, blob_id: next_blob_id as i32});
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
                        assert_eq!(donor_blob.recipient_blob, -1);
                        donor_blob.recipient_blob = recipient_blob_id;
                        let recipient_blob = &mut blobs.get_mut(&recipient_blob_id).unwrap();
                        recipient_blob.candidates.append(&mut donated_candidates);
                        break;
                    }
                    // donor_blob's candidates got merged to another blob.
                    assert_ne!(donor_blob.recipient_blob, -1);
                    if donor_blob.recipient_blob == recipient_blob_id {
                        break;  // Already merged it.
                    }
                    donor_blob_id = donor_blob.recipient_blob;
                }
            }
        }
    }
    // Return non-empty blobs. Note that the blob merging we just did will leave
    // some empty entries in the `blobs` mapping.
    let mut non_empty_blobs = Vec::<Blob>::new();
    for (_id, blob) in blobs {
        if !blob.candidates.is_empty() {
            assert_eq!(blob.recipient_blob, -1);
            debug!("got blob {:?}", blob);
            non_empty_blobs.push(blob);
        }
    }
    debug!("Found {} blobs in {:?}", non_empty_blobs.len(), blobs_start.elapsed());
    non_empty_blobs
}

/// Summarizes a star-like spot found by [get_stars_from_image()].
#[derive(Debug, Copy, Clone)]
pub struct StarDescription {
    /// Location of star centroid in image coordinates. (0.5, 0.5) corresponds
    /// to the center of the image's upper left pixel.
    pub centroid_x: f64,
    pub centroid_y: f64,

    /// Value of the brightest pixel in this star's region. Not background
    /// subtracted.
    pub peak_value: u8,

    /// Sum of the u8 pixel values of the star's region. The estimated
    /// background is subtracted.
    pub brightness: f64,

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
//
// image: This is either the original full resolution image, or a 2x2 or 4x4
//     binned image.
//
// full_res_image: Regardless of whether `image` is the original or a 2x2 or 4x4
//     binning, we arrange to do centroiding on the original resolution image.
//
// binning: 1 (no binning), 2, or 4.

fn gate_star_2d(
    blob: &Blob,
    image: &GrayImage,
    full_res_image: &GrayImage,
    binning: u32,
    noise_estimate: f64, sigma: f64,
    max_width: u32, max_height: u32) -> Option<StarDescription>
{
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
        image, &core, /*include_interior=*/true) {
        core_sum += i32::from(pixel_value);
        core_count += 1;
    }
    let core_mean = core_sum as f64 / core_count as f64;

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
            image, &core, /*include_interior=*/false) {
            outer_core_sum += i32::from(pixel_value);
            outer_core_count += 1;
        }
        let outer_core_mean = outer_core_sum as f64 / outer_core_count as f64;
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
        image, &neighbors, /*include_interior=*/false) {
        let is_corner =
            (x == neighbors.left() || x == neighbors.right()) &&
            (y == neighbors.top() || y == neighbors.bottom());
        if is_corner {
            continue;  // Exclude corner pixels.
        }
        neighbor_sum += i32::from(pixel_value);
        neighbor_count += 1;
    }
    let neighbor_mean = neighbor_sum as f64 / neighbor_count as f64;
    // Core average must be at least as bright as the neighbor average.
    if core_mean < neighbor_mean {
        debug!("Core average {} is less than neighbor average {} for blob {:?}",
               core_mean, neighbor_mean, core);
        return None;
    }

    // Compute average of pixels in next box out; this is one pixel inward from
    // the outer perimeter.
    let mut margin_sum: i32 = 0;
    let mut margin_count: i32 = 0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        image, &margin, /*include_interior=*/false) {
        margin_sum += i32::from(pixel_value);
        margin_count += 1;
    }
    let margin_mean = margin_sum as f64 / margin_count as f64;
    if core_mean <= margin_mean {
        debug!("Core average {} is not greater than margin average {} for blob {:?}",
               core_mean, margin_mean, core);
        return None;
    }

    // Gather information from the outer perimeter. These pixels represent the
    // background.
    let mut perimeter_sum: i32 = 0;
    let mut perimeter_count: i32 = 0;
    let mut perimeter_min = image.get_pixel(perimeter.left() as u32,
                                            perimeter.top() as u32).0[0];
    let mut perimeter_max = perimeter_min;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        image, &perimeter, /*include_interior=*/false) {
        perimeter_sum += i32::from(pixel_value);
        perimeter_count += 1;
        if pixel_value < perimeter_min {
            perimeter_min = pixel_value;
        } else if pixel_value > perimeter_max {
            perimeter_max = pixel_value;
        }
    }
    let background_est = perimeter_sum as f64 / perimeter_count as f64;
    debug!("background: {} for blob {:?}", background_est, core);

    // Compute a second noise estimate from the perimeter. If we're in clutter
    // such as an illuminated foreground object, this noise estimate will be
    // high, suppressing spurious "star" detections.
    let mut perimeter_dev_2: f64 = 0.0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        image, &perimeter, /*include_interior=*/false) {
        let res = i32::from(pixel_value) as f64 - background_est;
        perimeter_dev_2 += res * res;
    }
    let perimeter_stddev = (perimeter_dev_2 / perimeter_count as f64).sqrt();
    let max_noise_estimate = f64::max(noise_estimate, perimeter_stddev);

    // We require the perimeter pixels to be ~uniformly dark. See if any
    // perimeter pixel is too bright compared to the darkest perimeter
    // pixel.
    // The 3x sigma_noise threshold is empirically chosen to yield a low
    // rejection rate for actual sky background perimeter pixels.
    if (i32::from(perimeter_max) - i32::from(perimeter_min)) as f64 >
        3.0 * sigma * noise_estimate {
        debug!("Perimeter too varied for blob {:?}", core);
        return None;
    }

    // Verify that core average exceeds background by sigma * noise.
    if core_mean - background_est < sigma * max_noise_estimate {
        debug!("Core too weak for blob {:?}", core);
        return None;
    }

    // Star passes all of the 2d gates!

    let brightness;
    let num_saturated;
    let x;
    let y;
    let peak_value;
    if binning != 1 {
        // The `image` is binned. Compute moments using the full-res image.
        // Translate the margin (in the binned image) to the full-res image.
        let left = margin.left() as u32 * binning;
        let top = margin.top() as u32 * binning;
        let width = margin.width() * binning;
        let height = margin.height() * binning;
        let adj_width = cmp::min(left + width,
                                 full_res_image.width()) - left;
        let adj_height = cmp::min(top + height,
                                  full_res_image.height()) - top;
        let full_res_margin =
            Rect::at(left as i32, top as i32).of_size(adj_width, adj_height);
        (brightness, num_saturated, peak_value) =
            compute_brightness(full_res_image, &full_res_margin);
        (x, y) = compute_peak_coord(full_res_image, &full_res_margin);
    } else {
        (brightness, num_saturated, peak_value) =
            compute_brightness(full_res_image, &margin);
        (x, y) = compute_peak_coord(full_res_image, &margin);
    }
    Some(StarDescription{centroid_x: x + 0.5,
                         centroid_y: y + 0.5,
                         peak_value, brightness, num_saturated})
}

// Computes the background-subtracted brightness of the 2d image region.
// The outer perimeter of the bounding box is used for background
// estimation; the inner pixels of the bounding box are background
// subtracted and summed to form the brightness value.
// Returns: (summed pixel values, count of saturated pixels, peak pixel value)
fn compute_brightness(image: &GrayImage, bounding_box: &Rect) -> (f64, u16, u8) {
    let mut boundary_sum: i32 = 0;
    let mut boundary_count: i32 = 0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        image, bounding_box, /*include_interior=*/false) {
        boundary_sum += pixel_value as i32;
        boundary_count += 1;
    }
    let background_est = boundary_sum as f64 / boundary_count as f64;

    let inset = Rect::at(bounding_box.left() + 1, bounding_box.top() + 1)
        .of_size(bounding_box.width() - 2, bounding_box.height() - 2);

    let mut num_saturated = 0;
    let mut sum = 0.0;
    let mut peak_value: u8 = 0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        image, &inset, /*include_interior=*/true) {
        if pixel_value == 255_u8 {
            num_saturated += 1;
        }
        if pixel_value > peak_value {
            peak_value = pixel_value;
        }
        sum += pixel_value as f64 - background_est;
    }
    (f64::max(sum, 0.0), num_saturated, peak_value)
}

// Computes the position of the peak, with sub-pixel interpolation.
fn compute_peak_coord(image: &GrayImage, bounding_box: &Rect) -> (f64, f64) {
    let mut horizontal_projection = vec![0u32; bounding_box.width() as usize];
    let mut vertical_projection = vec![0u32; bounding_box.height() as usize];
    let x0 = bounding_box.left();
    let y0 = bounding_box.top();
    for (x, y, pixel_value) in EnumeratePixels::new(
        image, bounding_box, /*include_interior=*/true) {
        horizontal_projection[(x - x0) as usize] += pixel_value as u32;
        vertical_projection[(y - y0) as usize] += pixel_value as u32;
    }
    let peak_x = x0 as f64 + peak_coord_1d(horizontal_projection);
    let peak_y = y0 as f64 + peak_coord_1d(vertical_projection);
    (peak_x, peak_y)
}

fn peak_coord_1d(values: Vec<u32>) -> f64 {
    // We want a single peak value with lesser neighbors.
    let mut peak_val = 0;
    let mut peak_ind = 0;
    let mut in_run = false;
    let mut peak_run_length = 0;
    for (ind, val) in values.iter().enumerate() {
        if *val > peak_val || ind == 0 {
            peak_val = *val;
            peak_ind = ind;
            peak_run_length = 1;
            in_run = true;
        } else if *val == peak_val {
            if in_run {
                peak_run_length += 1;
            }
        } else {
            in_run = false;
        }
    }

    // If we have a run of equal-length values, just return the mid-coord of the
    // run. Yuck.
    if peak_run_length > 1 {
        return peak_ind as f64 + (peak_run_length - 1) as f64 / 2.0;
    }
    // If our peak is at either end of the vector, just return its coord. Yuck.
    if peak_ind == 0 || peak_ind == values.len() - 1 {
        return peak_ind as f64;
    }

    // We have a peak with two lesser neighbors. Apply quadratic interpolation.
    // https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    let a = values[peak_ind - 1] as f64;
    let b = values[peak_ind] as f64;
    let c = values[peak_ind + 1] as f64;
    let p = 0.5 * (a - c) / (a - 2.0 * b + c);
    assert!(p >= -0.5);
    assert!(p <= 0.5);

    peak_ind as f64 + p
}

/// Estimates the RMS noise of the given image. A small portion of the image
/// is processed as follows:
/// 1. The brightest pixels are excluded.
/// 2. The standard deviation of the remaining pixels is computed.
///
/// To guard against accidentally sampling a bright part of the image (moon?
/// streetlamp?), we sample a few image regions and choose the darkest one to
/// measure the noise.
pub fn estimate_noise_from_image(image: &GrayImage) -> f64 {
    let noise_start = Instant::now();
    let (width, height) = image.dimensions();

    // The IMX296mono camera on Raspberry Pi Zero 2 W has a noise problem that
    // causes some rows to differ in offset. We thus use horizontal cuts within
    // a single row when estimating noise, to avoid possible between-row level
    // variations.
    let cut_size = cmp::min(50, width / 4);

    // Sample three areas across the horizontal midline of the image.
    let mut stats_vec = vec![
        stats_for_roi(image, &Rect::at(
            (width/4 - cut_size/2) as i32, (height/2) as i32).of_size(cut_size, 1)),
        stats_for_roi(image, &Rect::at(
            (width*2/4 - cut_size/2) as i32, (height/2) as i32).of_size(cut_size, 1)),
        stats_for_roi(image, &Rect::at(
            (width*3/4 - cut_size/2) as i32, (height/2) as i32).of_size(cut_size, 1))
    ];
    // Pick the darkest cut by mean value.
    stats_vec.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
    let stddev = stats_vec[0].stddev;
    debug!("Noise estimate {} found in {:?}", stddev, noise_start.elapsed());
    stddev
}

/// Estimates the background and noise level of the given image region.
pub fn estimate_background_from_image_region(image: &GrayImage, roi: &Rect)
                                             -> (f64, f64) {
    let stats = stats_for_roi(image, roi);
    (stats.mean, stats.stddev)
}

// Returns mean/median/stddev for the given image region. Excludes bright
// pixels that are likely to be stars, as we're interested in the statistics
// of the sky background.
fn stats_for_roi(image: &GrayImage, roi: &Rect) -> HistogramStats {
    let mut histogram = [0_u32; 256];
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        image, roi, /*include_interior=*/true) {
        histogram[pixel_value as usize] += 1;
    }
    remove_stars_from_histogram(&mut histogram, /*sigma=*/8.0);
    stats_for_histogram(&histogram)
}

/// This function runs the CedarDetect algorithm on the supplied `image`, returning
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
///   `normalize_rows` Determines whether rows are normalized to have the same dark
///   level. Ignored if `binning` is 1.
///
///   `binning` 1 (no binning), 2 (2x2 binning), or 4 (4x4 binning). Specifies
///   whether `image` should be binned prior to star detction. Note that
///   computing the centroids of detected stars is always done in the full
///   resolution `image`.
///
///   `detect_hot_pixels` If true isolated hot pixels are detected and not
///   treated as stars. If false isolated hot pixels might be reported as stars.
///
///   `return_binned_image` If true, the 2x2 or 4x4 binning of `image` is
///   returned. Invalid if `binning` is 1.
///
/// # Returns
/// Vec<[StarDescription]>, in order of descending estimated brightness.
///
/// i32: The number of isolated hot pixels seen. See implementation for more
///   information about isolated hot pixels.
///
/// Option<GrayImage>: if `return_binned_image` is true, the 2x2 or 4x4 binning
///   of `image` is returned.
///
/// [u32; 256]: histogram of the `image` pixel values. Excludes the few leftmost
///   and rightmost columns. If `return_binned_image`, the histogram is over
///   that image rather than the input `image`.
pub fn get_stars_from_image(image: &GrayImage,
                            noise_estimate: f64,
                            sigma: f64,
                            normalize_rows: bool,
                            binning: u32,
                            detect_hot_pixels: bool,
                            return_binned_image: bool)
                            -> (Vec<StarDescription>,
                                /*hot_pixel_count*/i32,
                                Option<GrayImage>,
                                [u32; 256])
{
    match binning {
        1 => {
            if return_binned_image {
                panic!("cannot 'return_binned_image' when binning==1");
            }
        },
        2 | 4 => (),
        _ => {
            panic!("Invalid binning argument {}, must be 1, 2, or 4", binning);
        }
    }

    let start = Instant::now();

    // If noise estimate is below 0.25, assume that the image background has been
    // crushed to black; use a minimum noise value.
    let noise_estimate = f64::max(noise_estimate, 0.25);

    let mut stars = Vec::<StarDescription>::new();
    let mut hot_pixel_count = 0_i32;

    // CedarDetect clumps adjacent bright pixels to form a single star
    // candidate. The `max_size` value governs how large a clump can be before
    // it is rejected. Note that making `max_size` small can eliminate very
    // bright stars that "bleed" to many surrounding pixels.
    let max_size = image.dimensions().0 / 100;

    if binning == 1 {
        let (candidates_1d, histogram) =
            scan_image_for_candidates(image, noise_estimate, sigma,
                                      /*compute_histogram=*/true);
        let sigma_noise_2 = cmp::max((2.0 * sigma * noise_estimate + 0.5) as i16, 2);
        let mut filtered_candidates = Vec::<CandidateFrom1D>::new();
        for cand in candidates_1d {
            if !detect_hot_pixels {
                filtered_candidates.push(cand);
            } else if all_bright_are_hot(image, cand.x, cand.y, binning,
                                  sigma_noise_2)  {
                hot_pixel_count += 1;
            } else {
                filtered_candidates.push(cand);
            }
        }
        for blob in form_blobs_from_candidates(filtered_candidates) {
            if let Some(x) = gate_star_2d(&blob, image,
                               /*full_res_image=*/image,
                               binning, noise_estimate, sigma,
                               max_size, max_size) { stars.push(x) }
        }

        // Sort by brightness estimate, brightest first.
        stars.sort_by(|a, b| b.brightness.partial_cmp(&a.brightness).unwrap());

        debug!("Star detection completed in {:?}", start.elapsed());
        return (stars, hot_pixel_count, None, histogram.unwrap());
    }

    // We are binning by 2x or 4x.
    let mut binned_images = bin_and_histogram_2x2(image, normalize_rows);
    let mut binned_image = binned_images.binned;
    let mut histogram = binned_images.histogram;
    if binning == 4 {
        binned_images = bin_and_histogram_2x2(&binned_image, /*normalize_rows=*/false);
        binned_image = binned_images.binned;
        histogram = binned_images.histogram;
    }
    let noise_estimate_binned = f64::max(estimate_noise_from_image(&binned_image), 0.25);

    debug!("Image preprocessing completed in {:?}", start.elapsed());
    let detect_start = Instant::now();

    let sigma_noise_2 = cmp::max((2.0 * sigma * noise_estimate_binned + 0.5) as i16, 2);

    let (candidates_1d, _) =
        scan_image_for_candidates(&binned_image, noise_estimate_binned, sigma,
                                  /*compute_histogram=*/false);
    let mut filtered_candidates = Vec::<CandidateFrom1D>::new();
    for cand in candidates_1d {
        if !detect_hot_pixels {
            filtered_candidates.push(cand);
        } else if all_bright_are_hot(image, cand.x, cand.y, binning,
                              sigma_noise_2)  {
            hot_pixel_count += 1;
        } else {
            filtered_candidates.push(cand);
        }
    }
    for blob in form_blobs_from_candidates(filtered_candidates) {
        if let Some(x) = gate_star_2d(&blob, &binned_image,
                           /*full_res_image=*/image,
                           binning, noise_estimate_binned, sigma,
                           max_size/binning + 1, max_size/binning + 1) { stars.push(x) }
    }

    // Sort by brightness estimate, brightest first.
    stars.sort_by(|a, b| b.brightness.partial_cmp(&a.brightness).unwrap());

    debug!("Star detection completed in {:?}", detect_start.elapsed());
    (stars, hot_pixel_count, Some(binned_image), histogram)
}

// Given a star candidate in a (possibly) binned image, see if any pixel(s) in
// the full resolution image contribute to the star candidate are non-hot
// bright.
fn all_bright_are_hot(full_res_image: &GrayImage,
                      x: i32, y: i32, binning: u32,
                      sigma_noise_2: i16) -> bool {
    let width = full_res_image.dimensions().0 as i32;
    let image_pixels: &Vec<u8> = full_res_image.as_raw();

    // Translate coordinates to full resolution.
    let x_full = x * binning as i32;
    let y_full = y * binning as i32;

    // Process all of the pixels in the full_res_image that contribute
    // to the (binned) x/y pixel.
    for yi in 0..binning {
        let backing_y = y_full + yi as i32;
        let row_pixels: &[u8] = &image_pixels.as_slice()
            [(backing_y * width) as usize ..
             ((backing_y+1) * width) as usize];

        for xi in 0..binning {
            let backing_x = x_full + xi as i32;

            let start_x = backing_x - 3;
            let end_x = backing_x + 4;
            let gate = &row_pixels[start_x as usize .. end_x as usize];
            let (pixel_type, _) = classify_pixel(gate, sigma_noise_2);
            if pixel_type == PixelHotType::Bright {
                return false;
            }
        }
    }
    // All pixels are dark or hot.
    true
}

/// Summarizes an image region of interest. The pixel values used are not
/// background subtracted. Single hot pixels are replaced with interpolated
/// neighbor values when locating the peak pixel and when accumulating the
/// histogram.
#[derive(Debug)]
pub struct RegionOfInterestSummary {
    /// Histogram of pixel values in the ROI.
    pub histogram: [u32; 256],

    /// The location (in image coordinates) of the peak pixel. If there are
    /// multiple pixels with the peak value, the one closest to the image
    /// center is reported here. The application logic should use `histogram`
    /// to adjust exposure to avoid too many peak (saturated) pixels.
    pub peak_x: f64,
    pub peak_y: f64,

    /// Mean of the pixel values in a 3x3 pixel square centered on the peak
    /// pixel. Not dark subtracted.
    pub peak_value: f64,  // 0..255.
}

/// Gathers information from a region of interest of an image.
///
/// # Arguments
///   `image` - The image of which a portion will be summarized.
///   `roi` - Specifies the portion of `image` to be summarized.
///   `noise_estimate` The noise level of `image`. This is typically the noise
///       level returned by [estimate_noise_from_image()].
///   `sigma` - Specifies the statistical significance threshold used for
///       discriminating hot pixels from background. This can be the same value
///       as passed to [get_stars_from_image()].
///
/// # Returns
/// [RegionOfInterestSummary] Information that compactly provides some information
/// about the ROI. The `histogram` result can be used by an application's
/// auto-exposure logic; the `peak_x` and `peak_y` result can be used to identify
/// a target for focusing.
///
/// # Panics
/// The `roi` must exclude the three leftmost and three rightmost image columns.
/// The `roi` must exclude the three top and three bottom image rows.
pub fn summarize_region_of_interest(image: &GrayImage, roi: &Rect,
                                    noise_estimate: f64, sigma: f64)
                                    -> RegionOfInterestSummary {
    let process_roi_start = Instant::now();
    let (width, height) = image.dimensions();
    // Compute the ROI center, in image coordinates.
    let (roi_center_x, roi_center_y) =
        (roi.left() + roi.width() as i32 / 2,
         roi.top() + roi.height() as i32 / 2);
    let sigma_noise_2 = cmp::max((2.0 * sigma * noise_estimate + 0.5) as i16, 2);

    // Sliding gate needs to extend past left and right edges of ROI. Make sure
    // there's enough image.
    let gate_leftmost = roi.left() - 3;
    let gate_rightmost = roi.right() + 4;  // One past.
    assert!(gate_leftmost >= 0);
    assert!(gate_rightmost <= width as i32);

    // We also need top/bottom margin to allow centroiding.
    let top = roi.top() - 3;
    let bottom = roi.bottom() + 4;  // One past.
    assert!(top >= 0);
    assert!(bottom <= height as i32);
    let image_pixels: &Vec<u8> = image.as_raw();

    let mut histogram = [0_u32; 256];
    let mut peak_x = 0;
    let mut peak_y = 0;
    let mut peak_val = 0_u8;
    let mut peak_center_dist_sq = i32::MAX;
    for rownum in roi.top()..=roi.bottom() {
        // Get the slice of image_pixels corresponding to this row of the ROI.
        let row_start = (rownum * width as i32) as usize;
        let row_pixels: &[u8] = &image_pixels.as_slice()
            [row_start + gate_leftmost as usize ..
             row_start + gate_rightmost as usize];
        // Slide a 7 pixel gate across the row.
        let mut center_x = roi.left();
        for gate in row_pixels.windows(7) {
            let (pixel_type, val) = classify_pixel(gate, sigma_noise_2);
            if val >= peak_val {
                // See if this is a hot pixel.
                if pixel_type == PixelHotType::Hot {
                    // Hot pixel can't be peak. Substitute its value for the
                    // histogram.
                } else {
                    let center_dist_sq =
                        (center_x - roi_center_x) * (center_x - roi_center_x) +
                        (rownum - roi_center_y) * (rownum - roi_center_y);

                    let mut new_peak = false;
                    if val == peak_val {
                        // Break ties towards image center.
                        if center_dist_sq < peak_center_dist_sq {
                            new_peak = true;
                        }
                    } else {
                        new_peak = true;
                    }
                    if new_peak {
                        peak_x = center_x;
                        peak_y = rownum;
                        peak_val = val;
                        peak_center_dist_sq = center_dist_sq;
                    }
                }
            }  // Candidate peak.
            histogram[val as usize] += 1;
            center_x += 1;
        }
    }

    // Apply centroiding to get sub-pixel resolution for peak_x/y.
    let bounding_box = Rect::at(peak_x - 2, peak_y - 2).of_size(5, 5);
    let (mut x, mut y) = compute_peak_coord(image, &bounding_box);
    x += 0.5;
    y += 0.5;
    // Constrain x/y to be within ROI.
    x = cmp::max_by(x, roi.left() as f64, |a,b| a.partial_cmp(b).unwrap());
    x = cmp::min_by(x, roi.right() as f64, |a,b| a.partial_cmp(b).unwrap());
    y = cmp::max_by(y, roi.top() as f64, |a,b| a.partial_cmp(b).unwrap());
    y = cmp::min_by(y, roi.bottom() as f64, |a,b| a.partial_cmp(b).unwrap());

    let value_box = Rect::at(peak_x - 1, peak_y - 1).of_size(3, 3);
    let mut box_sum: i32 = 0;
    let mut box_count: i32 = 0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        image, &value_box, /*include_interior=*/true) {
        box_sum += pixel_value as i32;
        box_count += 1;
    }
    let peak_value = box_sum as f64 / box_count as f64;

    debug!("ROI processing completed in {:?}", process_roi_start.elapsed());
    RegionOfInterestSummary{histogram, peak_x: x, peak_y: y, peak_value}
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
        let mut stats = stats_for_roi(&image_5x4,
                                      &Rect::at(0, 0).of_size(5, 4));
        assert_eq!(stats.mean, 0_f64);
        assert_eq!(stats.median, 0);
        assert_eq!(stats.stddev, 0_f64);

        // Add some noise.
        image_5x4.put_pixel(1, 0, Luma::<u8>([1]));
        image_5x4.put_pixel(2, 0, Luma::<u8>([2]));
        image_5x4.put_pixel(3, 0, Luma::<u8>([1]));
        image_5x4.put_pixel(4, 0, Luma::<u8>([2]));
        image_5x4.put_pixel(0, 1, Luma::<u8>([1]));
        stats = stats_for_roi(&image_5x4,
                              &Rect::at(0, 0).of_size(5, 4));
        assert_eq!(stats.mean, 0.35);
        assert_eq!(stats.median, 0);
        assert_abs_diff_eq!(stats.stddev, 0.65, epsilon = 0.01);

        // Single bright pixel. This is removed because we eliminate the
        // brightest pixel(s).
        image_5x4.put_pixel(0, 0, Luma::<u8>([255]));
        stats = stats_for_roi(&image_5x4,
                              &Rect::at(0, 0).of_size(5, 4));
        assert_abs_diff_eq!(stats.mean, 0.368, epsilon = 0.001);
        assert_eq!(stats.median, 0);
        assert_abs_diff_eq!(stats.stddev, 0.66, epsilon = 0.01);

        // Add another non-zero pixel. This will be kept.
        image_5x4.put_pixel(0, 1, Luma::<u8>([4]));
        stats = stats_for_roi(&image_5x4,
                              &Rect::at(0, 0).of_size(5, 4));
        assert_abs_diff_eq!(stats.mean, 0.526, epsilon = 0.001);
        assert_eq!(stats.median, 0);
        assert_abs_diff_eq!(stats.stddev, 1.04, epsilon = 0.01);
    }

    #[test]
    fn test_estimate_noise_from_image() {
        let small_image = gaussian_noise(&GrayImage::new(100, 100),
                                         100.0, 3.0, 42);
        // The stddev is what we requested because there are no bright
        // outliers to discard.
        assert_abs_diff_eq!(estimate_noise_from_image(&small_image),
                            3.0, epsilon = 0.5);
        let large_image = gaussian_noise(&GrayImage::new(1000, 1000),
                                         100.0, 3.0, 42);
        assert_abs_diff_eq!(estimate_noise_from_image(&large_image),
                            3.0, epsilon = 0.5);
    }

    #[test]
    fn test_gate_star_1d_center_bright_wrt_background() {
        // Center minus background not bright enough.
        let mut gate: [u8; 7] = [10, 10, 10, 12, 10, 10, 10];
        let mut result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Center minus background is bright enough.
        gate = [10, 10, 11, 13, 11, 10, 10];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_center_bright_wrt_neighbor() {
        // Center is less than a neighbor.
        let mut gate: [u8; 7] = [10, 10, 11, 13, 14, 10, 10];
        let mut result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Ditto, other neighbor.
        gate = [10, 10, 14, 13, 11, 10, 10];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Center is at least as bright as its neighbors. Tie break is to
        // left (current candidate); this is explored further below.
        gate = [10, 10, 11, 13, 13, 10, 10];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_center_bright_wrt_margin() {
        // Center is not brighter than a margin.
        let mut gate: [u8; 7] = [10, 10, 11, 13, 11, 13, 10];
        let mut result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Ditto, other margin.
        gate = [10, 13, 11, 13, 11, 10, 10];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Center brighter than both margins.
        gate = [10, 12, 11, 13, 11, 12, 10];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_left_center_tie() {
        // When center and left have same value, tie is broken
        // based on next surrounding values.
        // In this case, the tie breaks to the left.
        let mut gate: [u8; 7] = [10, 11, 13, 13, 10, 11, 10];
        let mut result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Here, the tie breaks to the right (center pixel).
        gate = [10, 11, 13, 13, 11, 11, 10];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_center_right_tie() {
        // When center and right have same value, tie is broken
        // based on next surrounding values.
        // In this case, the tie breaks to the right.
        let mut gate: [u8; 7] = [10, 11, 11, 13, 13, 11, 10];
        let mut result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Here, the tie breaks to the left (center pixel).
        gate = [10, 11, 11, 13, 13, 10, 10];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_3=*/7);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_unequal_border() {
        // Border pixels differ too much, so candidate is rejected.
        let mut gate: [u8; 7] = [15, 10, 12, 18, 13, 10, 4];
        let mut result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_3=*/9);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Borders are close enough now.
        gate = [13, 10, 12, 18, 13, 10, 6];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_3=*/9);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_three_bright() {
        // Three equally bright pixels, left pixel.
        let mut gate: [u8; 7] = [10, 10, 10, 18, 18, 18, 10];
        let mut result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_3=*/9);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Middle pixel.
        gate = [10, 10, 18, 18, 18, 10, 10];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_3=*/9);
        assert_eq!(result_type, ResultType::Candidate);

        // Right pixel.
        gate = [10, 18, 18, 18, 10, 10, 10];
        result_type =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_3=*/9);
        assert_eq!(result_type, ResultType::Uninteresting);
    }

    #[test]
    fn test_form_blobs_from_candidates() {
        let mut candidates = Vec::<CandidateFrom1D>::new();
        // Candidates on the same row are not combined, even if they are close
        // together. This is because, in practice due to the operation of
        // gate_star_1d(), candidates in the same row will always be well
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
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1, 1.0, 6.0,
                           /*max_width=*/3, /*max_height=*/2) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1, 1.0, 6.0,
                           /*max_width=*/2, /*max_height=*/3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // We allow a 3x3 blob here.
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1, 1.0, 6.0,
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
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Too far to right.
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 4, y: 3});
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Too high.
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 3, y: 2});
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Too low.
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 3, y: 4});
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Just right!
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
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
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_9x9.put_pixel(4, 4, Luma::<u8>([20]));
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1, 1.0, 6.0, 3, 3) {
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
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_7x7.put_pixel(3, 3, Luma::<u8>([14]));
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
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
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_7x7.put_pixel(3, 3, Luma::<u8>([15]));
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
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
       26, 10, 12, 15, 12, 10,  9;
        9, 10, 15, 30, 15, 10,  9;
        9, 10, 12, 15, 12, 10,  9;
        9, 10, 10, 10, 10, 10,  9;
        9,  9,  9,  9,  9,  9,  9);
        // Perimeter has an anomalously bright pixel.
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Repair the perimeter.
        image_7x7.put_pixel(0, 2, Luma::<u8>([12]));
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
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
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_7x7.put_pixel(3, 3, Luma::<u8>([14]));
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1, 1.0, 6.0, 3, 3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate"),
        }
    }

    #[test]
    fn test_peak_coord_1d() {
        // All zeroes.
        let values = vec![0, 0, 0, 0, 0];
        assert_eq!(peak_coord_1d(values), 2.0);

        // All same value.
        let values = vec![10, 10, 10, 10, 10];
        assert_eq!(peak_coord_1d(values), 2.0);

        // Run of two peak values. Disjoint peak value later in vector
        // does not extend the run.
        let values = vec![1, 2, 2, 1, 2];
        assert_eq!(peak_coord_1d(values), 1.5);

        // Peak is left-most position.
        let values = vec![12, 2, 2, 1, 2];
        assert_eq!(peak_coord_1d(values), 0.0);

        // Peak is right-most position.
        let values = vec![12, 2, 2, 1, 22];
        assert_eq!(peak_coord_1d(values), 4.0);

        // Quadratic interpolation cases.
        let values = vec![1, 3, 10, 3, 2];
        assert_eq!(peak_coord_1d(values), 2.0);
        let values = vec![1, 3, 10, 5, 2];
        assert_abs_diff_eq!(peak_coord_1d(values), 2.1, epsilon = 0.05);
        let values = vec![1, 9, 10, 0, 2];
        assert_abs_diff_eq!(peak_coord_1d(values), 1.6, epsilon = 0.05);
    }

    #[test]
    fn test_brightness() {
        let image_7x7 = gray_image!(
        9,  9,  9,   9,  9,  9,  9;
        9, 10, 10,  10, 10, 10,  9;
        9, 10, 12, 255, 12, 10,  9;
        9, 10, 14, 255, 14, 10,  9;
        9, 10, 12,  14, 30, 10,  9;
        9, 10, 10,  10, 10, 10,  9;
        9,  9,  9,   9,  9,  9,  9);
        let neighbors = Rect::at(1, 1).of_size(5, 5);
        let (brightness, num_sat, peak_value) = compute_brightness(&image_7x7, &neighbors);
        assert_abs_diff_eq!(brightness,
                            528.0, epsilon = 0.1);
        assert_eq!(num_sat, 2);
        assert_eq!(peak_value, 255);
    }

    #[test]
    #[should_panic]
    fn test_summarize_region_of_interest_left_edge() {
        let roi = Rect::at(2, 3).of_size(3, 2);
        let image_9x9 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9);
        // Cannot give ROI too close to left edges.
        let _roi_summary = summarize_region_of_interest(
            &image_9x9, &roi, 1.0, 5.0);
    }

    #[test]
    #[should_panic]
    fn test_summarize_region_of_interest_right_edge() {
        let roi = Rect::at(4, 3).of_size(3, 2);
        let image_9x9 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9);
        // Cannot give ROI too close to right edges.
        let _roi_summary = summarize_region_of_interest(
            &image_9x9, &roi, 1.0, 5.0);
    }

    #[test]
    fn test_summarize_region_of_interest_good_edges() {
        let roi = Rect::at(3, 3).of_size(3, 2);
        // 80 is a hot pixel, is replaced by interpolation of its left
        // and right neighbors.
        let image_9x9 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  7,  80, 9,  9,  9,  9;
            9,  9,  9,  11, 20, 32, 10, 9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9);
        // ROI is correct distance from left+right edges.
        let roi_summary = summarize_region_of_interest(
            &image_9x9, &roi, 1.0, 5.0);
        assert_eq!(roi_summary.histogram[7], 1);
        assert_eq!(roi_summary.histogram[8], 1);
        assert_eq!(roi_summary.histogram[9], 1);
        assert_eq!(roi_summary.histogram[11], 1);
        assert_eq!(roi_summary.histogram[20], 1);
        assert_eq!(roi_summary.histogram[32], 1);
        // The hot pixel is not the peak, but it can influence
        // the centroided position of the peak and the peak_value.
        assert_abs_diff_eq!(roi_summary.peak_x,
                            4.5, epsilon = 0.1);
        assert_abs_diff_eq!(roi_summary.peak_y,
                            3.7, epsilon = 0.1);
        assert_abs_diff_eq!(roi_summary.peak_value,
                            20.8, epsilon = 0.1);
    }

    #[test]
    fn test_summarize_region_of_interest_peak_upper_left() {
        let roi = Rect::at(3, 3).of_size(3, 2);
        let image_9x9 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  99, 99, 69, 9,  9,  9,  9,  9;
            9,  99, 80, 60, 50,  9,  9,  9,  9;
            9,  69, 60, 50, 40, 10, 9,  9,  9;
            9,  9,  50, 40, 20, 10, 9, 9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9);
        let roi_summary = summarize_region_of_interest(
            &image_9x9, &roi, 1.0, 5.0);
        assert_abs_diff_eq!(roi_summary.peak_x,
                            3.0, epsilon = 0.01);
        assert_abs_diff_eq!(roi_summary.peak_y,
                            3.0, epsilon = 0.01);
    }

    #[test]
    fn test_summarize_region_of_interest_peak_lower_right() {
        let roi = Rect::at(3, 3).of_size(3, 2);
        let image_9x9 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9, 20, 40, 50,  9,  9;
            9,  9,  9,  9, 40, 50, 60,  9,  9;
            9,  9,  9,  9, 50, 60, 80,  9,  9;
            9,  9,  9,  9, 59, 69, 89,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9);
        let roi_summary = summarize_region_of_interest(
            &image_9x9, &roi, 1.0, 5.0);
        assert_abs_diff_eq!(roi_summary.peak_x,
                            5.0, epsilon = 0.01);
        assert_abs_diff_eq!(roi_summary.peak_y,
                            4.0, epsilon = 0.01);
    }

    #[test]
    fn test_summarize_region_of_interest_peak_tiebreak_to_center() {
        let roi = Rect::at(3, 3).of_size(5, 5);
        let image_11x11 = gray_image!(
            9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9, 11,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9, 11,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9, 11,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9;
            9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9);
        let roi_summary = summarize_region_of_interest(
            &image_11x11, &roi, 1.0, 5.0);
        assert_abs_diff_eq!(roi_summary.peak_x,
                            5.5, epsilon = 0.01);
        assert_abs_diff_eq!(roi_summary.peak_y,
                            5.5, epsilon = 0.01);
    }

}  // mod tests.
