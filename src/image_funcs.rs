// Copyright (c) 2025 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

use image::GrayImage;
use std::sync::OnceLock;
use crate::histogram_funcs::estimate_dark_level;

pub struct Binned2x2Result {
    pub binned: GrayImage,
    pub histogram: [u32; 256],
}

pub type BinnerFn = fn(&GrayImage, bool) -> Binned2x2Result;

static BINNER_FN: OnceLock<BinnerFn> = OnceLock::new();

pub fn set_binner(func: BinnerFn) {
    log::info!("Setting image preprocessing function.");
    let _ = BINNER_FN.set(func);  // Ignores error if already set.
}

// `normalize_rows` Determines whether rows are normalized to have the same dark
//     level. See IMX296mono notes below.
pub fn bin_and_histogram_2x2(image: &GrayImage, normalize_rows: bool) -> Binned2x2Result {
    match BINNER_FN.get() {
        Some(f) => f(image, normalize_rows),
        None => bin_and_histogram_2x2_default(image, normalize_rows),
    }
}

// Default implementation, used if set_binner() was not called.
fn bin_and_histogram_2x2_default(image: &GrayImage, normalize_rows: bool)
                                -> Binned2x2Result
{
    let normalized;
    let source_image = if normalize_rows {
        normalized = apply_row_normalization(image);
        &normalized
    } else {
        image
    };
    let (width, height) = source_image.dimensions();

    // 2x2 box filter.
    let new_width = width / 2;
    let new_height = height / 2;
    let mut resized_image = Vec::with_capacity((new_width * new_height) as usize);
    let mut histogram = [0u32; 256];

    let source_pixels = source_image.as_raw();

    for y in (0..height & !1).step_by(2) {  // Ensure even height bound
        for x in (0..width & !1).step_by(2) {   // Ensure even width bound
            // Get 2x2 block.
            let p1 = source_pixels[(y * width + x) as usize] as u16;
            let p2 = source_pixels[(y * width + x + 1) as usize] as u16;
            let p3 = source_pixels[((y + 1) * width + x) as usize] as u16;
            let p4 = source_pixels[((y + 1) * width + x + 1) as usize] as u16;

            // Average the 2x2 block.
            let avg = ((p1 + p2 + p3 + p4) / 4) as u8;

            resized_image.push(avg);
            histogram[avg as usize] += 1;
        }
    }

    let output_image = GrayImage::from_raw(
        new_width, new_height, resized_image).unwrap();
    Binned2x2Result { binned: output_image, histogram }
}

// The IMX296mono camera on Raspberry Pi Zero 2 W has a noise problem that
// causes some rows to differ in offset. We estimate the dark level for each row
// and equalize all rows to the same 'bias' dark level during the binning
// process.
fn apply_row_normalization(image: &GrayImage) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut normalized_pixels = Vec::with_capacity((width * height) as usize);
    let source_pixels = image.as_raw();

    for y in 0..height {
        // Build histogram for this row.
        let mut row_histogram = [0u32; 256];
        let row_start = (y * width) as usize;
        let row_end = ((y + 1) * width) as usize;

        for &pixel in &source_pixels[row_start..row_end] {
            row_histogram[pixel as usize] += 1;
        }

        // Get estimated dark level for this row.
        let row_dark_level =
            estimate_dark_level(&row_histogram, width as usize);
        let bias = 2.0;
        let adjust = (bias - row_dark_level).round() as i16;

        // Normalize row pixels.
        for &pixel in &source_pixels[row_start..row_end] {
            let adjusted = (pixel as i16) + adjust;
            let normalized = adjusted.clamp(0, 255) as u8;
            normalized_pixels.push(normalized);
        }
    }

    GrayImage::from_raw(width, height, normalized_pixels).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_bin_and_histogram_2x2() {
        // Create a 4x4 test image.
        let mut img = GrayImage::new(4, 4);

        // Fill with known values.
        for y in 0..4 {
            for x in 0..4 {
                img.put_pixel(x, y, Luma([((y * 4 + x) as u8 + 1)]));
            }
        }

        let result = bin_and_histogram_2x2(&img, /*normalize_rows=*/false);
        assert_eq!(result.binned.dimensions(), (2, 2));

        // Check the sums and histogram.
        // Top-left 2x2: [1,2,5,6] -> sum = 14, avg = 3
        assert_eq!(result.binned.get_pixel(0, 0)[0], 3);

        // Top-right 2x2: [3,4,7,8] -> sum = 22, avg = 5
        assert_eq!(result.binned.get_pixel(1, 0)[0], 5);

        // Bottom-left 2x2: [9,10,13,14] -> sum = 46, avg = 11
        assert_eq!(result.binned.get_pixel(0, 1)[0], 11);

        // Bottom-right 2x2: [11,12,15,16] -> sum = 54, avg = 13
        assert_eq!(result.binned.get_pixel(1, 1)[0], 13);

        // Check histogram: should have 1 pixel each at values 3, 5, 11, 13
        assert_eq!(result.histogram[3], 1);
        assert_eq!(result.histogram[5], 1);
        assert_eq!(result.histogram[11], 1);
        assert_eq!(result.histogram[13], 1);

        // All other histogram bins should be 0
        let total_pixels: u32 = result.histogram.iter().sum();
        assert_eq!(total_pixels, 4); // 2x2 output image
    }
}
