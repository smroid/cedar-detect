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
//! * Automatically classifies and rejects hot pixels.
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
//! such as [Tetra3](https://github.com/esa/tetra3). It can also be incorporated
//! into satellite star trackers.
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
//! pixels. CedarDetect can thus be confused when stars are too closely spaced, or
//! a star is close to a hot pixel. Such situations will usually cause closely
//! spaced stars to fail to be detected. Note that for applications such as
//! plate solving, this is probably for the better.
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
//!     mis-classify stars as hot pixels. A simple remedy is to slightly defocus
//!     causing stars to occupy a central peak pixel with a bit of spread into
//!     immediately adjacent pixels.
//! * CedarDetect does not tolerate more than maybe a single pixel of motion blur.
//!
//! ## Centroid estimation
//!
//! CedarDetect's computes a sub-pixel centroid position for each detected star
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

use image::{GrayImage, ImageBuffer, Luma, Primitive};
use imageproc::rect::Rect;
use log::{debug};

// When get_stars_from_image() is called with the 'use_binned_image' option,
// an intermediate 2x2 binned image is formed by summing pixel values. This
// results in up to 10 bit pixel values, so we use a u16 ImageBuffer when
// processing the 2x2 binned image (rather than discarding information by
// scaling back down to 8 bit pixel values).
type Gray16Image = ImageBuffer<Luma<u16>, Vec<u16>>;

// An iterator over the pixels of a region of interest. Yields pixels in raster
// scan order.
// P: u8 or u16.
struct EnumeratePixels<'a, P: Primitive> {
    image: &'a ImageBuffer<Luma<P>, Vec<P>>,
    roi: &'a Rect,
    include_interior: bool,

    // Identifies the next pixel to be yielded. If cur_y is beyond the ROI's
    // bottom, the iteration is finished.
    cur_x: i32,
    cur_y: i32,
}

impl<'a, P: Primitive> EnumeratePixels<'a, P> {
    // If include_interior is false, only the perimeter is enumerated.
    fn new(image: &'a ImageBuffer<Luma<P>, Vec<P>>, roi: &'a Rect, include_interior: bool)
           -> EnumeratePixels<'a, P> {
        let (width, height) = image.dimensions();
        assert!(roi.left() >= 0);
        assert!(roi.top() >= 0);
        assert!(roi.right() < width as i32);
        assert!(roi.bottom() < height as i32);
        EnumeratePixels{image, roi, include_interior,
                        cur_x: roi.left(), cur_y: roi.top()}
    }
}

// P: u8 or u16.
impl<'a, P: Primitive> Iterator for EnumeratePixels<'a, P> {
    type Item = (i32, i32, P);  // x, y, pixel value.

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
// The `detect_hot_pixels` argument determines whether hot pixel
// detection/substitution is done.
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

// P: u8 (original) or u16 (2x2 binned).
fn gate_star_1d<P: Primitive + std::fmt::Debug>(
    gate: &[P], sigma_noise_2: i16, sigma_noise_1_5: i16,
    detect_hot_pixels: bool)
    -> (/*corrected_value*/P, ResultType)
where u16: From<P>
{
    debug_assert!(sigma_noise_2 > 0);
    debug_assert!(sigma_noise_1_5 > 0);
    debug_assert!(sigma_noise_1_5 <= sigma_noise_2);
    // Examining assembler output suggests that fetching these exterior pixels
    // first eliminates bounds checks on the interior pixels. I would have
    // thought that the compiler would do this... in any case, the measured
    // performance doesn't seem to change.
    let lb = u16::from(gate[0]) as i16;
    let rb = u16::from(gate[6]) as i16;

    let lm = u16::from(gate[1]) as i16;
    let l = u16::from(gate[2]) as i16;
    let c = u16::from(gate[3]) as i16;
    let r = u16::from(gate[4]) as i16;
    let rm = u16::from(gate[5]) as i16;

    // Center pixel must be sigma * estimated noise brighter than the estimated
    // background. Do this test first, because it eliminates the vast majority
    // of candidates.
    let est_background_2 = lb + rb;
    let center_minus_background_2 = c + c - est_background_2;
    if center_minus_background_2 < sigma_noise_2 {
        return (gate[3], ResultType::Uninteresting);
    }
    // Center pixel must be at least as bright as its immediate left/right
    // neighbors.
    if l > c || c < r {
        return (gate[3], ResultType::Uninteresting);
    }
    // Center pixel must be strictly brighter than its left/right margin.
    if lm >= c || c <= rm {
        return (gate[3], ResultType::Uninteresting);
    }
    if l == c {
        // Break tie between left and center.
        if lm > r {
            // Left will have been the center of its own candidate entry.
            return (gate[3], ResultType::Uninteresting);
        }
    }
    if c == r {
        // Break tie between center and right.
        if l <= rm {
            // Right will be the center of its own candidate entry.
            return (gate[3], ResultType::Uninteresting);
        }
    }
    // Average of l+r (minus background) must exceed 0.25 * center (minus
    // background).
    if detect_hot_pixels {
        let sum_neighbors_minus_background = l + r - est_background_2;
        if 4 * sum_neighbors_minus_background <= center_minus_background_2 {
            // For ROI processing purposes, replace the hot pixel with its
            // neighbors' value.
            return (P::from((l + r) / 2).unwrap(), ResultType::HotPixel);
        }
    }
    // We require the border pixels to be ~uniformly dark. See if there is too
    // much brightness difference between the border pixels.
    // The 2x sigma_noise threshold is empirically chosen to yield a low
    // rejection rate for actual sky background border pixels.
    let border_diff = (lb - rb).abs();
    if border_diff > sigma_noise_2 {
        return (gate[3], ResultType::Uninteresting);
    }
    // We have a candidate star from our 1d analysis!
    debug!("candidate: {:?}", gate);
    return (gate[3], ResultType::Candidate);
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
// `detect_hot_pixels` Determines whether hot pixel detection/substitution is
//     done.
// `create_binned_image10` If true, a 2x2 binning of `image` is returned. Note
//     that hot pixels are replaced during the binning process. Binning is by
//     summing, and so the result pixels are 2 bits wider than the input.
// `create_binned_image8` If true, a 2x2 binning of `image` is returned. Note
//     that hot pixels are replaced during the binning process. For binning is
//     by averaging, so the result pixels are 8 bits. Note that is OK for this
//     to be true if `create_binned_image10` is false.

// Returns:
// Vec<CandidateFrom1D>: the identifed star candidates, in raster scan order.
// i32: count of hot pixels detected.
// Option<Gray16Image>: if `create_binned_image10` is true, the 10 bit binned
//   image is returned here.
// Option<GrayImage>: if `create_binned_image8` is true, the 8 bit binned image
//   is returned here.
//
// P: u8 (original) or u16 (2x2 binned).
fn scan_image_for_candidates<P: Primitive + std::fmt::Debug>(
    image: &ImageBuffer<Luma<P>, Vec<P>>,
    noise_estimate: f32, sigma: f32,
    detect_hot_pixels: bool,
    create_binned_image10: bool,
    create_binned_image8: bool)
    -> (Vec<CandidateFrom1D>,
        /*hot_pixel_count*/i32,
        Option<Gray16Image>,
        Option<GrayImage>)
where u16: From<P>
{
    let row_scan_start = Instant::now();
    let mut hot_pixel_count = 0;
    let width = image.dimensions().0 as usize;
    let height = image.dimensions().1 as usize;
    let image_pixels: &Vec<P> = image.as_raw();
    let mut candidates = Vec::<CandidateFrom1D>::new();
    let sigma_noise_2 = cmp::max((2.0 * sigma * noise_estimate + 0.5) as i16, 2);
    let sigma_noise_1_5 = cmp::max((1.5 * sigma * noise_estimate + 0.5) as i16, 1);

    let (binned_width, binned_height) = (width / 2, height / 2);
    let mut binned_image10_data = Vec::<u16>::new();
    let mut binned_image8_data = Vec::<u8>::new();
    if create_binned_image10 || create_binned_image8 {
        // Allocate uninitialized storage to receive the 2x2 binned image data.
        let num_binned_pixels = binned_width * binned_height;
        binned_image10_data.reserve(num_binned_pixels as usize);
        unsafe { binned_image10_data.set_len(num_binned_pixels) }
        if create_binned_image8 {
            binned_image8_data.reserve(num_binned_pixels as usize);
            unsafe { binned_image8_data.set_len(num_binned_pixels) }
        }
    }

    for rownum in 0..height {
        // Get the slice of image_pixels corresponding to this row.
        let row_pixels: &[P] = &image_pixels.as_slice()
            [rownum * width .. (rownum+1) * width];
        let mut center_x = 2_usize;
        // We duplicate the loops, slightly different according to whether we
        // are binning. We do this to avoid having a conditional on binning
        // in the inner loop.
        if create_binned_image10 || create_binned_image8 {
            if rownum == height-1 && height & 1 != 0 {
                break;  // Skip final row on odd-height image when binning.
            }
            let binned_rownum = rownum / 2;
            // Set up accumulator row for binning.
            let binned_slice: &mut[u16] = &mut binned_image10_data.as_mut_slice()
                [binned_rownum * binned_width .. (binned_rownum+1) * binned_width];
            if rownum & 1 == 0 {
                binned_slice.fill(0);
            }
            binned_slice[0] += u16::from(row_pixels[0]);
            binned_slice[0] += u16::from(row_pixels[1]);
            binned_slice[1] += u16::from(row_pixels[2]);
            // Slide a 7 pixel gate across the row.
            for gate in row_pixels.windows(7) {
                center_x += 1;
                let (pixel_value, result_type) = gate_star_1d(
                    gate, sigma_noise_2, sigma_noise_1_5, detect_hot_pixels);
                binned_slice[center_x / 2] += u16::from(pixel_value);
                match result_type {
                    ResultType::Uninteresting => (),
                    ResultType::Candidate => {
                        candidates.push(CandidateFrom1D{x: center_x as i32,
                                                        y: rownum as i32});
                    },
                    ResultType::HotPixel => { hot_pixel_count += 1; },
                }
            }
            center_x += 1;
            while center_x < width - 1 {
                binned_slice[center_x / 2] += u16::from(row_pixels[center_x]);
                center_x += 1;
            }
            if width & 1 == 0 {
                // For even width, include final pixel in row.
                binned_slice[center_x / 2] += u16::from(row_pixels[center_x]);
            }
            if rownum & 1 != 0 {
                let binned_rownum = rownum / 2;
                if create_binned_image8 {
                    let binned_image10_row: &mut[u16] = &mut binned_image10_data.as_mut_slice()
                        [binned_rownum * binned_width .. (binned_rownum+1) * binned_width];
                    let binned_image8_row: &mut[u8] = &mut binned_image8_data.as_mut_slice()
                        [binned_rownum * binned_width .. (binned_rownum+1) * binned_width];
                    for x in 0..binned_width {
                        binned_image8_row[x] = (binned_image10_row[x] / 4) as u8;
                    }
                }
            }
        } else {
            // Slide a 7 pixel gate across the row.
            for gate in row_pixels.windows(7) {
                center_x += 1;
                let (_pixel_value, result_type) = gate_star_1d(
                    gate, sigma_noise_2, sigma_noise_1_5, detect_hot_pixels);
                match result_type {
                    ResultType::Uninteresting => (),
                    ResultType::Candidate => {
                        candidates.push(CandidateFrom1D{x: center_x as i32,
                                                        y: rownum as i32});
                    },
                    ResultType::HotPixel => { hot_pixel_count += 1; },
                }
            }
        }
    }
    debug!("Image scan found {} candidates and {} hot pixels in {:?}",
          candidates.len(), hot_pixel_count, row_scan_start.elapsed());
    let mut binned10_result: Option<Gray16Image> = None;
    let mut binned8_result: Option<GrayImage> = None;
    if create_binned_image10 {
        binned10_result = Some(Gray16Image::from_raw(
            binned_width as u32, binned_height as u32,
            binned_image10_data).unwrap());
    }
    if create_binned_image8 {
        binned8_result = Some(GrayImage::from_raw(
            binned_width as u32, binned_height as u32,
            binned_image8_data).unwrap());
    }
    (candidates, hot_pixel_count, binned10_result, binned8_result)
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
    pub centroid_x: f32,
    pub centroid_y: f32,

    /// Sum of the u8 pixel values of the star's region. The estimated
    /// background is subtracted.
    pub brightness: f32,

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
//
// image: This is either the original full resolution image (8 bits), or a
//     2x2 binned image (10 bits).
//
// full_res_image: Regardless of whether `image` is the original or a 2x2
//     binning, we arrange to do centroiding on the original resolution image.
//
// P: u8 (original) or u16 (2x2 binned).
fn gate_star_2d<P: Primitive>(
    blob: &Blob,
    image: &ImageBuffer<Luma<P>, Vec<P>>,
    full_res_image: &GrayImage,
    noise_estimate: f32, sigma: f32,
    detect_hot_pixels: bool,
    max_width: u32, max_height: u32) -> Option<StarDescription>
where i32: From<P>
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
        &image, &core, /*include_interior=*/true) {
        core_sum += i32::from(pixel_value);
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
            outer_core_sum += i32::from(pixel_value);
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
        neighbor_sum += i32::from(pixel_value);
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
        margin_sum += i32::from(pixel_value);
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
    let mut perimeter_min = image.get_pixel(perimeter.left() as u32,
                                            perimeter.top() as u32).0[0];
    let mut perimeter_max = perimeter_min;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        &image, &perimeter, /*include_interior=*/false) {
        perimeter_sum += i32::from(pixel_value);
        perimeter_count += 1;
        if pixel_value < perimeter_min {
            perimeter_min = pixel_value;
        }
        if pixel_value > perimeter_max {
            perimeter_max = pixel_value;
        }
    }
    let background_est = perimeter_sum as f32 / perimeter_count as f32;
    debug!("background: {} for blob {:?}", background_est, core);

    // Compute a second noise estimate from the perimeter. If we're in clutter
    // such as an illuminated foreground object, this noise estimate will be
    // high, suppressing spurious "star" detections.
    let mut perimeter_dev_2: f32 = 0.0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        &image, &perimeter, /*include_interior=*/false) {
        let res = i32::from(pixel_value) as f32 - background_est;
        perimeter_dev_2 += res * res;
    }
    let perimeter_stddev = (perimeter_dev_2 / perimeter_count as f32).sqrt();
    let max_noise_estimate = f32::max(noise_estimate, perimeter_stddev);

    // We require the perimeter pixels to be ~uniformly dark. See if any
    // perimeter pixel is too bright compared to the darkest perimeter
    // pixel.
    // The 2x sigma_noise threshold is empirically chosen to yield a low
    // rejection rate for actual sky background perimeter pixels.
    if (i32::from(perimeter_max) - i32::from(perimeter_min)) as f32 >
        2.0 * sigma * noise_estimate {
        debug!("Perimeter too varied for blob {:?}", core);
        return None;
    }

    // Verify that core average exceeds background by sigma * noise.
    if core_mean - background_est < sigma * max_noise_estimate {
        debug!("Core too weak for blob {:?}", core);
        return None;
    }
    if detect_hot_pixels && core_width == 1 && core_height == 1 {
        // Verify that the neighbor average (minus background) exceeds 0.25 *
        // core (minus background).
        if neighbor_mean - background_est <= 0.25 * (core_mean - background_est) {
            // Hot pixel.
            debug!("Neighbors too weak for blob {:?}", core);
            return None;
        }
    }

    // Star passes all of the 2d gates!

    let brightness;
    let num_saturated;
    let x;
    let y;
    if image_width < full_res_image.dimensions().0 {
        // The `image` is binned. Compute moments using the full-res image.
        // Translate the margin (in the binned image) to the full-res image.
        let left: u32 = margin.left() as u32 * 2;
        let top = margin.top() as u32 * 2;
        let width = margin.width() * 2;
        let height = margin.height() * 2;
        let adj_width = cmp::min(left + width,
                                 full_res_image.width()) - left;
        let adj_height = cmp::min(top + height,
                                  full_res_image.height()) - top;
        let full_res_margin =
            Rect::at(left as i32, top as i32).of_size(adj_width, adj_height);
        (brightness, num_saturated) = compute_brightness(full_res_image, &full_res_margin);
        (x, y) = compute_peak_coord(full_res_image, &full_res_margin);
    } else {
        (brightness, num_saturated) = compute_brightness(full_res_image, &margin);
        (x, y) = compute_peak_coord(full_res_image, &margin);
    }
    Some(StarDescription{centroid_x: x + 0.5,
                         centroid_y: y + 0.5,
                         brightness, num_saturated})
}

// Computes the background-subtracted brightness of the 2d image region.
// The outer perimeter of the bounding box is used for background
// estimation; the inner pixels of the bounding box are background
// subtracted and summed to form the brightness value.
// Returns: (summed pixel values, count of saturated pixels)
fn compute_brightness(image: &GrayImage, bounding_box: &Rect) -> (f32, u16) {
    let mut boundary_sum: i32 = 0;
    let mut boundary_count: i32 = 0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        &image, &bounding_box, /*include_interior=*/false) {
        boundary_sum += pixel_value as i32;
        boundary_count += 1;
    }
    let background_est = boundary_sum as f32 / boundary_count as f32;

    let inset = Rect::at(bounding_box.left() + 1, bounding_box.top() + 1)
        .of_size(bounding_box.width() - 2, bounding_box.height() - 2);

    let mut num_saturated = 0;
    let mut sum = 0.0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        &image, &inset, /*include_interior=*/true) {
        if pixel_value == 255_u8 {
            num_saturated += 1;
        }
        sum += pixel_value as f32 - background_est;
    }
    (f32::max(sum, 0.0), num_saturated)
}

// Computes the position of the peak, with sub-pixel interpolation.
fn compute_peak_coord(image: &GrayImage, bounding_box: &Rect) -> (f32, f32) {
    let mut horizontal_projection = vec![0u32; bounding_box.width() as usize];
    let mut vertical_projection = vec![0u32; bounding_box.height() as usize];
    let x0 = bounding_box.left();
    let y0 = bounding_box.top();
    for (x, y, pixel_value) in EnumeratePixels::new(
        &image, &bounding_box, /*include_interior=*/true) {
        horizontal_projection[(x - x0) as usize] += pixel_value as u32;
        vertical_projection[(y - y0) as usize] += pixel_value as u32;
    }
    let peak_x = x0 as f32 + peak_coord_1d(horizontal_projection);
    let peak_y = y0 as f32 + peak_coord_1d(vertical_projection);
    (peak_x, peak_y)
}

fn peak_coord_1d(values: Vec<u32>) -> f32 {
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
        return peak_ind as f32 + (peak_run_length - 1) as f32 / 2.0;
    }
    // If our peak is at either end of the vector, just return its coord. Yuck.
    if peak_ind == 0 || peak_ind == values.len() - 1 {
        return peak_ind as f32;
    }

    // We have a peak with two lesser neighbors. Apply quadratic interpolation.
    // https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    let a = values[peak_ind - 1] as f32;
    let b = values[peak_ind] as f32;
    let c = values[peak_ind + 1] as f32;
    let p = 0.5 * (a - c) / (a - 2.0 * b + c);
    assert!(p >= -0.5);
    assert!(p <= 0.5);

    peak_ind as f32 + p
}

/// Estimates the RMS noise of the given image. A small portion of the image
/// is processed as follows:
/// 1. The brightest pixels are excluded.
/// 2. The standard deviation of the remaining pixels is computed.
///
/// To guard against accidentally sampling a bright part of the image (moon?
/// streetlamp?), we sample a few image regions and choose the darkest one to
/// measure the noise.
// P: u8 or u16.
pub fn estimate_noise_from_image<P: Primitive>(
    image: &ImageBuffer<Luma<P>, Vec<P>>) -> f32
where usize: From<P>
{
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
// P: u8 or u16.
fn stats_for_roi<P: Primitive>(image: &ImageBuffer<Luma<P>, Vec<P>>,
                               roi: &Rect) -> (/*median*/f32, /*stddev*/f32)
where usize: From<P>,
{
    // P is either 8 bits or 10 bits (2x2 binned values). Size histogram
    // accordingly.
    let mut histogram: [u32; 1024] = [0; 1024];
    let mut pixel_count = 0;
    for (_x, _y, pixel_value) in EnumeratePixels::new(
        &image, &roi, /*include_interior=*/true) {
        histogram[usize::from(pixel_value)] += 1;
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
    for h in 0_usize..1024 {
        if h >= star_cutoff {
            histogram[h] = 0;
        }
    }
    debug!("De-starred histogram: {:?}", histogram);
    let (_mean, stddev, median) = stats_for_histogram(&histogram);
    (median as f32, stddev)
}

fn trim_histogram(histogram: &[u32; 1024], count_to_keep: u32)
                  -> [u32; 1024] {
    let mut trimmed_histogram = *histogram;
    let mut count = 0;
    for h in 0..1024 {
        let bin_count = trimmed_histogram[h];
        if count + bin_count > count_to_keep {
            let excess = count + bin_count - count_to_keep;
            trimmed_histogram[h] -= excess;
        }
        count += trimmed_histogram[h];
    }
    return trimmed_histogram;
}

fn stats_for_histogram(histogram: &[u32; 1024])
                       -> (/*mean*/f32, /*stddev*/f32, /*median*/usize) {
    let mut count = 0;
    let mut first_moment = 0;
    for h in 0..1024 {
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
    for h in 0..1024 {
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
///   `max_size` - CedarDetect clumps adjacent bright pixels to form a single star
///   candidate. The `max_size` argument governs how large a clump can be before
///   it is rejected. Note that making `max_size` small can eliminate very
///   bright stars that "bleed" to many surrounding pixels. `max_size` is always
///   given in full-resolution units, even if `use_binned_image` is true.
///   Typical `max_size` value: 8.
///
///   `use_binned_image` If true a 2x2 binning of `image` (with hot pixels
///   removed) is used for star detection. Note that computing the centroids of
///   detected stars is always done in the full resolution `image`.
///
///   `return_binned_image` If true, a 2x2 binning of `image` is returned. Note
///   that hot pixels are replaced during the binning process.
///
/// # Returns
/// Vec<[StarDescription]>, in order of descending brightness.
///
/// i32: The number of hot pixels seen. See implementation for more information
/// about hot pixels.
///
/// Option<GrayImage>: if `return_binned_image` is true, the 2x2 binning of `image`
///   is returned.
pub fn get_stars_from_image(
    image: &GrayImage,
    noise_estimate: f32, sigma: f32, max_size: u32,
    use_binned_image: bool,
    return_binned_image: bool)
    -> (Vec<StarDescription>, /*hot_pixel_count*/i32, Option<GrayImage>) {
    // If noise estimate is below 0.5, assume that the image background has been
    // crushed to black; use a minimum noise value.
    let noise_estimate8 = f32::max(noise_estimate, 0.5);

    let mut binned_image10: Option<Gray16Image> = None;
    let mut binned_image8: Option<GrayImage> = None;
    let mut hot_pixel_count = 0;
    if use_binned_image || return_binned_image {
        (_, hot_pixel_count, binned_image10, binned_image8) =
            scan_image_for_candidates(image, noise_estimate8, sigma,
                                      /*detect_hot_pixels=*/true,
                                      /*create_binned_image10=*/true,
                                      /*create_binned_image8=*/return_binned_image);
    }

    let mut stars = Vec::<StarDescription>::new();
    if use_binned_image {
        let binned_noise_estimate = estimate_noise_from_image(
            &binned_image10.as_ref().unwrap());
        // Use a higher noise floor for 10-bit binned image.
        let noise_estimate10 = f32::max(binned_noise_estimate, 1.5);
        let (candidates, _, _, _) =
            scan_image_for_candidates(&binned_image10.as_ref().unwrap(),
                                      noise_estimate10, sigma,
                                      /*detect_hot_pixels=*/false,
                                      /*create_binned_image10=*/false,
                                      /*create_binned_image8=*/false);
        for blob in form_blobs_from_candidates(candidates) {
            match gate_star_2d(&blob, &binned_image10.as_ref().unwrap(),
                               /*full_res_image=*/image,
                               noise_estimate10, sigma,
                               /*detect_hot_pixels=*/false,
                               max_size/2 + 1, max_size/2 + 1) {
                Some(x) => stars.push(x),
                None => ()
            }
        }
    } else {
        let (candidates, local_hot_pixel_count, _, _) =
            scan_image_for_candidates(image, noise_estimate8, sigma,
                                      /*detect_hot_pixels=*/true,
                                      /*create_binned_image10=*/false,
                                      /*create_binned_image8=*/false);
        hot_pixel_count = local_hot_pixel_count;
        for blob in form_blobs_from_candidates(candidates) {
            match gate_star_2d(&blob, image,
                               /*full_res_image=*/image,
                               noise_estimate8, sigma,
                               /*detect_hot_pixels=*/true,
                               max_size, max_size) {
                Some(x) => stars.push(x),
                None => ()
            }
        }
    }

    // Sort by brightness estimate, brightest first.
    stars.sort_by(|a, b| b.brightness.partial_cmp(&a.brightness).unwrap());

    (stars, hot_pixel_count, binned_image8)
}

/// Summarizes an image region of interest. The pixel values used are not
/// background subtracted. Single hot pixels are replaced with interpolated
/// neighbor values when locating the peak pixel and when accumulating the
/// histogram.
#[derive(Debug)]
pub struct RegionOfInterestSummary {
    /// Histogram of pixel values in the ROI.
    pub histogram: [u32; 256],

    /// The location (in image coordinates) of the peak pixel (after correcting
    /// for hot pixels). If there are multiple pixels with the peak value, it is
    /// unspecified which one's location is reported here. The application logic
    /// should use `histogram` to adjust exposure to avoid too many peak
    /// (saturated) pixels.
    pub peak_x: f32,
    pub peak_y: f32,
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
/// auto-exposure logic; the `peak_x` and `peak_y` result can be used to identify
/// a target for focusing.
///
/// # Panics
/// The `roi` must exclude the three leftmost and three rightmost image columns.
/// The `roi` must exclude the three top and three bottom image rows.
pub fn summarize_region_of_interest(image: &GrayImage, roi: &Rect,
                                    noise_estimate: f32, sigma: f32)
                                    -> RegionOfInterestSummary {
    let process_roi_start = Instant::now();

    let mut peak_x = 0;
    let mut peak_y = 0;
    let mut peak_val = 0_u8;
    let (width, height) = image.dimensions();
    // Sliding gate needs to extend past left and right edges of ROI. Make sure
    // there's enough image.
    let gate_leftmost: i32 = roi.left() as i32 - 3;
    let gate_rightmost = roi.right() + 4;  // One past.
    assert!(gate_leftmost >= 0);
    assert!(gate_rightmost <= width as i32);
    // We also need top/bottom margin to allow centroiding.
    let top: i32 = roi.top() as i32 - 3;
    let bottom = roi.bottom() + 4;  // One past.
    assert!(top >= 0);
    assert!(bottom <= height as i32);
    let image_pixels: &Vec<u8> = image.as_raw();

    let mut histogram: [u32; 256] = [0_u32; 256];

    // We'll replace hot image pixels and do centroiding in the cleaned up
    // image.
    let mut cleaned_image = image.clone();

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
                gate_star_1d(gate, sigma_noise_2, sigma_noise_1_5,
                             /*detect_hot_pixels=*/true);
            cleaned_image.put_pixel(center_x as u32, rownum as u32,
                                    Luma::<u8>([pixel_value]));
            histogram[pixel_value as usize] += 1;
            if pixel_value >= peak_val {
                peak_x = center_x;
                peak_y = rownum;
                peak_val = pixel_value;
            }
            center_x += 1;
        }
    }

    // Apply centroiding to get sub-pixel resolution for peak_x/y.
    let bounding_box = Rect::at(peak_x - 3, peak_y - 3).of_size(7, 7);
    let (x, y) = compute_peak_coord(&cleaned_image, &bounding_box);
    debug!("ROI processing completed in {:?}", process_roi_start.elapsed());
    RegionOfInterestSummary{histogram,
                            peak_x: x + 0.5,
                            peak_y: y + 0.5}
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
        let mut histogram = [0_u32; 1024];
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
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 12);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Center minus background is bright enough.
        gate = [10, 10, 11, 13, 11, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_center_bright_wrt_neighbor() {
        // Center is less than a neighbor.
        let mut gate: [u8; 7] = [10, 10, 11, 13, 14, 10, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Ditto, other neighbor.
        gate = [10, 10, 14, 13, 11, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Center is at least as bright as its neighbors. Tie break is to
        // left (current candidate); this is explored further below.
        gate = [10, 10, 11, 13, 13, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_center_bright_wrt_margin() {
        // Center is not brighter than a margin.
        let mut gate: [u8; 7] = [10, 10, 11, 13, 11, 13, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Ditto, other margin.
        gate = [10, 13, 11, 13, 11, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Center brighter than both margins.
        gate = [10, 12, 11, 13, 11, 12, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
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
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Here, the tie breaks to the right (center pixel).
        gate = [10, 11, 13, 13, 11, 11, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
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
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Here, the tie breaks to the left (center pixel).
        gate = [10, 11, 11, 13, 13, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/3, false);
        assert_eq!(value, 13);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_hot_pixel() {
        // Neighbors are too dark, so bright center is deemed a hot pixel.
        let mut gate: [u8; 7] = [10, 10, 10, 15, 12, 10, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4, true);
        assert_eq!(value, 11);
        assert_eq!(result_type, ResultType::HotPixel);

        // Neighbors have enough brightness, so bright center is deemed a
        // star candidate.
        gate = [10, 10, 12, 15, 12, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/2, true);
        assert_eq!(value, 15);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_hot_pixel_disabled() {
        // As abovbe, except hot pixel processing is disabled.
        let mut gate: [u8; 7] = [10, 10, 10, 15, 12, 10, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4, false);
        assert_eq!(value, 15);  // Hot pixel not substituted.
        assert_eq!(result_type, ResultType::Candidate);

        // Neighbors have enough brightness, so bright center is deemed a
        // star candidate.
        gate = [10, 10, 12, 15, 12, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/5, /*sigma_noise_1_5=*/2, false);
        assert_eq!(value, 15);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_unequal_border() {
        // Border pixels differ too much, so candidate is rejected.
        let mut gate: [u8; 7] = [14, 10, 12, 18, 13, 10, 5];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4, false);
        assert_eq!(value, 18);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Borders are close enough now.
        gate = [13, 10, 12, 18, 13, 10, 6];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4, false);
        assert_eq!(value, 18);
        assert_eq!(result_type, ResultType::Candidate);
    }

    #[test]
    fn test_gate_star_1d_three_bright() {
        // Three equally bright pixels, left pixel.
        let mut gate: [u8; 7] = [10, 10, 10, 18, 18, 18, 10];
        let (mut value, mut result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4, false);
        assert_eq!(value, 18);
        assert_eq!(result_type, ResultType::Uninteresting);

        // Middle pixel.
        gate = [10, 10, 18, 18, 18, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4, false);
        assert_eq!(value, 18);
        assert_eq!(result_type, ResultType::Candidate);

        // Right pixel.
        gate = [10, 18, 18, 18, 10, 10, 10];
        (value, result_type) =
            gate_star_1d(&gate, /*sigma_noise_2=*/7, /*sigma_noise_1_5=*/4, false);
        assert_eq!(value, 18);
        assert_eq!(result_type, ResultType::Uninteresting);
    }

    #[test]
    fn test_create_binned_image() {
        // Test the create_binned_image10 aspect of scan_image_for_candidates().
        // Even dimensions, no hot pixel.
        let image_12x4 = gray_image!(
           10, 20, 30, 40, 50, 60, 70, 80, 90,100,110,120;
            2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24;
            0,  0,  0,  1,  0,  1,  1,  0,  1,  1,  1,  1;
            0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  3);
        let (candidates, hot_pixel_count, binned_image10, _) =
            scan_image_for_candidates(&image_12x4, 1.0, 8.0,
                                      /*detect_hot_pixels=*/true,
                                      /*create_binned_image10=*/true,
                                      /*create_binned_image8=*/false);
        assert_eq!(candidates.len(), 0);
        assert_eq!(hot_pixel_count, 0);
        let binned_image10 = binned_image10.unwrap();
        assert_eq!(binned_image10.width(), 6);
        assert_eq!(binned_image10.height(), 2);
        let expected_binned10 = gray_image!(type: u16,
            36,  84, 132, 180, 228, 276;
             0,   1,   2,   3,   4,   6);
        assert_eq!(binned_image10, expected_binned10);

        // Odd dimensions, with hot pixel.
        let image_11x3 = gray_image!(
           10, 20, 30, 40, 50, 60, 70, 80, 90,100,110;
            2,  4,  6,250, 10, 12, 14, 16, 18, 20, 22;
            0,  0,  0,  1,  0,  1,  1,  0,  1,  1,  1);
        let (candidates, hot_pixel_count, binned_image10, _) =
            scan_image_for_candidates(&image_11x3, 1.0, 8.0,
                                      /*detect_hot_pixels=*/true,
                                      /*create_binned_image10=*/true,
                                      /*create_binned_image8=*/false);
        assert_eq!(candidates.len(), 0);
        assert_eq!(hot_pixel_count, 1);
        let binned_image10 = binned_image10.unwrap();
        assert_eq!(binned_image10.width(), 5);
        assert_eq!(binned_image10.height(), 1);
        let expected_binned10 = gray_image!(type: u16,
                                            36,  84, 132, 180, 228);
        assert_eq!(binned_image10, expected_binned10);

        // Same, except with hot pixel detection turned off. Also request
        // 8 bit version of binned image.
        let (candidates, hot_pixel_count, binned_image10, binned_image8) =
            scan_image_for_candidates(&image_11x3, 1.0, 8.0,
                                      /*detect_hot_pixels=*/false,
                                      /*create_binned_image10=*/true,
                                      /*create_binned_image8=*/true);
        assert_eq!(candidates.len(), 1);
        assert_eq!(hot_pixel_count, 0);
        let binned_image10 = binned_image10.unwrap();
        assert_eq!(binned_image10.width(), 5);
        assert_eq!(binned_image10.height(), 1);
        let expected_binned10 = gray_image!(type: u16,
                                            36, 326, 132, 180, 228);
        assert_eq!(binned_image10, expected_binned10);

        let binned_image8 = binned_image8.unwrap();
        assert_eq!(binned_image8.width(), 5);
        assert_eq!(binned_image8.height(), 1);
        let expected_binned8 = gray_image!(9, 81, 33, 45, 57);
        assert_eq!(binned_image8, expected_binned8);
    }

    #[test]
    fn test_form_blobs_from_candidates() {
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
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1.0, 6.0, true,
                           /*max_width=*/3, /*max_height=*/2) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1.0, 6.0, true,
                           /*max_width=*/2, /*max_height=*/3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // We allow a 3x3 blob here.
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1.0, 6.0, true,
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
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Too far to right.
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 4, y: 3});
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Too high.
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 3, y: 2});
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Too low.
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 3, y: 4});
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => ()
        }
        // Just right!
        blob.candidates.clear();
        blob.candidates.push(CandidateFrom1D{x: 3, y: 3});
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
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
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_9x9.put_pixel(4, 4, Luma::<u8>([20]));
        match gate_star_2d(&blob, &image_9x9, &image_9x9, 1.0, 6.0, true, 3, 3) {
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
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_7x7.put_pixel(3, 3, Luma::<u8>([14]));
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
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
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_7x7.put_pixel(3, 3, Luma::<u8>([15]));
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
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
       22, 10, 12, 15, 12, 10,  9;
        9, 10, 15, 30, 15, 10,  9;
        9, 10, 12, 15, 12, 10,  9;
        9, 10, 10, 10, 10, 10,  9;
        9,  9,  9,  9,  9,  9,  9);
        // Perimeter has an anomalously bright pixel.
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Repair the perimeter.
        image_7x7.put_pixel(0, 2, Luma::<u8>([12]));
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
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
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Make center bright enough.
        image_7x7.put_pixel(3, 3, Luma::<u8>([14]));
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
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
        // Neighbor ring is not brighter enough compared to core (core is a hot
        // pixel).
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
            Some(_star_description) => panic!("Expected rejection"),
            None => (),
        }
        // Same, but turn off hot pixel detection.
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0,
                           /*detect_hot_pixels=*/false, 3, 3) {
            Some(_star_description) => (),
            None => panic!("Expected candidate"),
        }
        // Enable hot pixel detection, and make neighbors bright enough.
        image_7x7.put_pixel(2, 2, Luma::<u8>([12]));
        image_7x7.put_pixel(3, 2, Luma::<u8>([12]));
        match gate_star_2d(&blob, &image_7x7, &image_7x7, 1.0, 6.0, true, 3, 3) {
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
        let (brightness, num_sat) = compute_brightness(&image_7x7, &neighbors);
        assert_abs_diff_eq!(brightness,
                            528.0, epsilon = 0.1);
        assert_eq!(num_sat, 2);
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
            9,  9,  9,  7,  80,  9, 9,  9,  9;
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
        // The hot pixel is not the peak.
        assert_abs_diff_eq!(roi_summary.peak_x,
                            5.37, epsilon = 0.01);
        assert_abs_diff_eq!(roi_summary.peak_y,
                            4.52, epsilon = 0.01);
    }

}  // mod tests.
