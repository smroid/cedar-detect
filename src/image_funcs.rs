// Copyright (c) 2025 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

use image::GrayImage;
use std::sync::OnceLock;

pub struct Binned2x2Result {
    pub binned: GrayImage,
    pub histogram: [u32; 256],
}

pub type BinAndHistoFn = fn(&GrayImage) -> Binned2x2Result;
pub type Bin2x2Fn = fn(&GrayImage) -> GrayImage;

static BIN_AND_HISTO_FN: OnceLock<BinAndHistoFn> = OnceLock::new();
static BIN2X2_FN: OnceLock<Bin2x2Fn> = OnceLock::new();

pub fn set_binner(bin_and_histo: BinAndHistoFn, bin2x2: Bin2x2Fn) {
    log::info!("Setting image binning functions.");
    let _ = BIN_AND_HISTO_FN.set(bin_and_histo); // Ignores error if already set.
    let _ = BIN2X2_FN.set(bin2x2); // Ignores error if already set.
}

pub fn bin_and_histogram_2x2(image: &GrayImage) -> Binned2x2Result {
    match BIN_AND_HISTO_FN.get() {
        Some(f) => f(image),
        None => bin_and_histogram_2x2_default(image),
    }
}

pub fn bin_2x2(image: &GrayImage) -> GrayImage {
    match BIN2X2_FN.get() {
        Some(f) => f(image),
        None => bin_2x2_default(image),
    }
}

// Default implementation, used if set_binner() was not called.
fn bin_2x2_default(image: &GrayImage) -> GrayImage {
    let (width, height) = image.dimensions();
    let new_width = width / 2;
    let new_height = height / 2;
    let mut resized_image = Vec::with_capacity((new_width * new_height) as usize);
    let source_pixels = image.as_raw();
    for y in (0..height & !1).step_by(2) {
        for x in (0..width & !1).step_by(2) {
            let p1 = source_pixels[(y * width + x) as usize] as u16;
            let p2 = source_pixels[(y * width + x + 1) as usize] as u16;
            let p3 = source_pixels[((y + 1) * width + x) as usize] as u16;
            let p4 = source_pixels[((y + 1) * width + x + 1) as usize] as u16;
            resized_image.push(((p1 + p2 + p3 + p4) / 4) as u8);
        }
    }
    GrayImage::from_raw(new_width, new_height, resized_image).unwrap()
}

// Default implementation, used if set_binner() was not called.
fn bin_and_histogram_2x2_default(image: &GrayImage) -> Binned2x2Result
{
    let (width, height) = image.dimensions();

    // 2x2 box filter.
    let new_width = width / 2;
    let new_height = height / 2;
    let mut resized_image = Vec::with_capacity((new_width * new_height) as usize);
    let mut histogram = [0u32; 256];

    let source_pixels = image.as_raw();

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

        let result = bin_and_histogram_2x2(&img);
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

    #[test]
    fn test_bin_2x2() {
        // Same 4x4 image as test_bin_and_histogram_2x2.
        let mut img = GrayImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                img.put_pixel(x, y, Luma([((y * 4 + x) as u8 + 1)]));
            }
        }

        let result = bin_2x2(&img);
        assert_eq!(result.dimensions(), (2, 2));
        assert_eq!(result.get_pixel(0, 0)[0], 3);   // [1,2,5,6]    -> 3
        assert_eq!(result.get_pixel(1, 0)[0], 5);   // [3,4,7,8]    -> 5
        assert_eq!(result.get_pixel(0, 1)[0], 11);  // [9,10,13,14] -> 11
        assert_eq!(result.get_pixel(1, 1)[0], 13);  // [11,12,15,16] -> 13

        // Result must match bin_and_histogram_2x2.
        let full = bin_and_histogram_2x2(&img);
        assert_eq!(result.as_raw(), full.binned.as_raw());
    }
}
