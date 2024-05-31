// Copyright (c) 2023 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

use std::cmp::{max, min};

pub struct HistogramStats {
    pub mean: f32,
    pub median: usize,
    pub stddev: f32,
}

pub fn stats_for_histogram(histogram: &[u32]) -> HistogramStats {
    let mut count = 0;
    let mut first_moment = 0;
    for h in 0..histogram.len() {
        let bin_count = histogram[h];
        count += bin_count;
        first_moment += bin_count * h as u32;
    }
    if count == 0 {
        return HistogramStats{mean: 0.0, median: 0, stddev: 0.0};
    }
    let mean = first_moment as f32 / count as f32;
    let mut second_moment: f32 = 0.0;
    let mut sub_count = 0;
    let mut median = 0;
    for h in 0..histogram.len() {
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
    HistogramStats{mean, median, stddev}
}

/// Return the histogram bin number N such that the total number of bin entries
/// at or below N exceeds `fraction` * the total number of bin entries over the
/// entire histogram.
pub fn get_level_for_fraction(histogram: &[u32], fraction: f32) -> usize {
    assert!(fraction >= 0.0);
    assert!(fraction <= 1.0);
    let mut count = 0;
    for h in 0..histogram.len() {
        count += histogram[h];
    }
    let goal = (fraction * count as f32) as u32;
    count = 0;
    for h in 0..histogram.len() {
        count += histogram[h];
        if count >= goal {
            return h;
        }
    }
    assert!(false);  // Should not get here.
    return 0;
}

/// Return the average of the N highest histogram entry values.
pub fn average_top_values(histogram: &[u32], num_top_values: usize) -> u8 {
    let mut accum_count = 0;
    let mut accum_val: u32 = 0;
    for bin in (1..256).rev() {
        let remain = num_top_values - accum_count;
        if remain == 0 {
            break;
        }
        let count = min(histogram[bin], remain as u32);
        accum_val += bin as u32 * count;
        accum_count += count as usize;
    }
    max(accum_val / accum_count as u32, 1) as u8
}

/// Updates `histogram`, removing counts deemed to be contributed by stars. We
/// assume that the histogram was obtained from an image consisting of well
/// focused stars on a mostly dark background.
/// We form an estimate of the background noise and use this to remove counts
/// from histogram bins of pixel values greater than `sigma` times the noise
/// estimate.
/// What remains is deemed to be the histogram of the non-star image pixels.
pub fn remove_stars_from_histogram(histogram: &mut [u32], sigma: f32) {
    let mut pixel_count = 0;
    for h in 0..histogram.len() {
        pixel_count += histogram[h];
    }
    // Do a sloppy trim of the brightest pixels; this will give us a (probably
    // overly) de-starred mean and stddev that we can use for a more precise
    // trim.
    let mut copied_histogram: Vec<u32> = Vec::new();
    copied_histogram.extend_from_slice(histogram);
    trim_histogram(&mut copied_histogram, pixel_count * 9 / 10);
    let stats = stats_for_histogram(&copied_histogram);
    // Any pixel whose value is sigma * stddev above the mean is deemed a star and
    // kicked out of the histogram.
    let star_cutoff = (stats.mean as f32 +
                       sigma * f32::max(stats.stddev, 1.0)) as usize;
    for h in 0..histogram.len() {
        if h >= star_cutoff {
            histogram[h] = 0;
        }
    }
}

fn trim_histogram(histogram: &mut [u32], count_to_keep: u32) {
    let mut count = 0;
    for h in 0..histogram.len() {
        let bin_count = histogram[h];
        if count + bin_count > count_to_keep {
            let excess = count + bin_count - count_to_keep;
            histogram[h] -= excess;
        }
        count += histogram[h];
    }
}


#[cfg(test)]
mod tests {
    use crate::histogram_funcs::stats_for_histogram;

    #[test]
    fn test_stats_for_histogram() {
        let mut histogram = [0_u32; 1024];
        histogram[10] = 2;
        histogram[20] = 2;
        let stats = stats_for_histogram(&histogram);
        assert_eq!(stats.mean, 15.0);
        assert_eq!(stats.median, 10);
        assert_eq!(stats.stddev, 5.0);
    }

}  // mod tests.
