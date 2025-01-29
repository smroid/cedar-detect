// Copyright (c) 2024 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

use std::cmp::{max, min};

pub struct HistogramStats {
    pub mean: f64,
    pub median: usize,
    pub stddev: f64,
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
    let mean = first_moment as f64 / count as f64;
    let mut second_moment: f64 = 0.0;
    let mut sub_count = 0;
    let mut median = 0;
    for h in 0..histogram.len() {
        let bin_count = histogram[h];
        second_moment += bin_count as f64 * (h as f64 - mean) * (h as f64 - mean);
        if sub_count < count / 2 {
            sub_count += bin_count;
            if sub_count >= count / 2 {
                median = h;
            }
        }
    }
    let stddev = (second_moment / count as f64).sqrt();
    HistogramStats{mean, median, stddev}
}

/// Given a histogram of a row's u8 data, and the width of the row, return
/// an estimate of the row's dark level (u8 range, as f32).
pub fn estimate_row_dark_level(row_histogram: &[usize; 256], width: usize) -> f32 {
    // Method: exclude the bottom 1% of values. Compute the mean of the next
    // 1% of values.
    let one_percent = width / 100;
    if one_percent == 0 {
        // Too little data; just return the lowest non-zero bin.
        for h in 0..row_histogram.len() {
            if row_histogram[h] > 0 {
                return h as f32;
            }
        }
    }
    // Skip first 'one_percent' of non-zero values.
    let mut skip_count = one_percent;
    let mut cur_bin = 0;
    let mut count_remaining = 0_usize;
    for h in 0..row_histogram.len() {
        let bin_count = row_histogram[h];
        if bin_count == 0 {
            continue;
        }
        if bin_count < skip_count {
            skip_count -= bin_count;
            continue;
        }
        cur_bin = h;
        count_remaining = bin_count - skip_count;
        break;
    }
    // We've skipped the first 'one_percent'. Start accumulating
    // the next 'one_percent'.
    if count_remaining >= one_percent {
        return cur_bin as f32;
    }
    let mut accum = cur_bin * count_remaining;
    cur_bin += 1;
    let mut accum_count = one_percent - count_remaining;

    for h in cur_bin..row_histogram.len() {
        let bin_count = row_histogram[h];
        if bin_count < accum_count {
            accum += h * bin_count;
            accum_count -= bin_count;
            continue;
        }
        accum += h * accum_count;
        break;
    }

    accum as f32 / one_percent as f32
}

/// Shift the histogram bins by the indicated amount.
pub fn shift_histogram(histogram: &mut[usize], shift: i32) {
    let num_bins = histogram.len();
    if num_bins <= 1 {
        return;
    }
    if shift > 0 {
        let shift = min(shift as usize, num_bins-1);
        // Move histogram bins to the right. At the top of the histogram,
        // accumulate incoming bin values. At the bottom of the histogram,
        // zero the vacated bins.
        let top_bin = num_bins - 1;
        for i in 1..=shift {
            histogram[top_bin] += histogram[top_bin - i];
        }
        let mut bin = top_bin - 1;
        while bin >= shift {
            histogram[bin] = histogram[bin - shift];
            bin -= 1;
        }
        for i in 0..shift {
            histogram[i] = 0;
        }
    }
    if shift < 0 {
        let shift = min(-shift as usize, num_bins-1);
        // Move histogram bins to the left. At the bottom of the histogram,
        // accumulate incoming bin values. At the top of the histogram,
        // zero the vacated bins.
        for i in 1..=shift {
            histogram[0] += histogram[i];
        }
        let mut bin = 1;
        while bin < num_bins - shift {
            histogram[bin] = histogram[bin + shift];
            bin += 1;
        }
        for i in 1..=shift {
            histogram[num_bins - i] = 0;
        }
    }
}

/// Return the histogram bin number N such that the total number of bin entries
/// at or below N exceeds `fraction` * the total number of bin entries over the
/// entire histogram.
pub fn get_level_for_fraction(histogram: &[u32], fraction: f64) -> usize {
    assert!(fraction >= 0.0);
    assert!(fraction <= 1.0);
    let mut count = 0;
    for h in 0..histogram.len() {
        count += histogram[h];
    }
    let goal = (fraction * count as f64) as u32;
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
    if accum_count == 0 {
        0
    } else {
        max(accum_val / accum_count as u32, 1) as u8
    }
}

/// Updates `histogram`, removing counts deemed to be contributed by stars. We
/// assume that the histogram was obtained from an image consisting of well
/// focused stars on a mostly dark background.
/// We form an estimate of the background noise and use this to remove counts
/// from histogram bins of pixel values greater than `sigma` times the noise
/// estimate.
/// What remains is deemed to be the histogram of the non-star image pixels.
pub fn remove_stars_from_histogram(histogram: &mut [u32], sigma: f64) {
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
    let star_cutoff = (stats.mean +
                       sigma * f64::max(stats.stddev, 1.0)) as usize;
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
    use crate::histogram_funcs::{estimate_row_dark_level, shift_histogram,
                                 stats_for_histogram};

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

    #[test]
    fn test_estimate_row_dark_level_small_width() {
        let mut row_histogram = [0_usize; 256];

        // Width is small, so 1% count underflows.
        row_histogram[2] = 1;
        row_histogram[3] = 9;
        assert_eq!(estimate_row_dark_level(&row_histogram, 10), 2.0);
    }

    #[test]
    fn test_estimate_row_dark_level_same_bin_after_skip() {
        let mut row_histogram = [0_usize; 256];

        row_histogram[2] = 5;  // Skipped.
        row_histogram[4] = 15;  // First 5 skipped.
        row_histogram[10] = 980;
        assert_eq!(estimate_row_dark_level(&row_histogram, 1000), 4.0);
    }

    #[test]
    fn test_estimate_row_dark_level_next_bins_after_skip() {
        let mut row_histogram = [0_usize; 256];

        row_histogram[2] = 5;  // Skipped.
        row_histogram[4] = 5;  // Skipped.
        row_histogram[6] = 5;
        row_histogram[8] = 985;
        assert_eq!(estimate_row_dark_level(&row_histogram, 1000), 7.0);
    }

    #[test]
    fn test_shift_single_bin_histogram() {
        let mut histogram = [1];
        shift_histogram(&mut histogram, 0);
        assert_eq!(histogram[0], 1);
        shift_histogram(&mut histogram, 1);
        assert_eq!(histogram[0], 1);
        shift_histogram(&mut histogram, 2);
        assert_eq!(histogram[0], 1);
        shift_histogram(&mut histogram, -1);
        assert_eq!(histogram[0], 1);
        shift_histogram(&mut histogram, -2);
        assert_eq!(histogram[0], 1);
    }

    #[test]
    fn test_shift_histogram() {
        // Shift right by varying amounts.
        let mut histogram = [1, 2, 3, 4];
        shift_histogram(&mut histogram, 0);
        assert_eq!(histogram[0], 1);
        assert_eq!(histogram[1], 2);
        assert_eq!(histogram[2], 3);
        assert_eq!(histogram[3], 4);

        shift_histogram(&mut histogram, 1);
        assert_eq!(histogram[0], 0);
        assert_eq!(histogram[1], 1);
        assert_eq!(histogram[2], 2);
        assert_eq!(histogram[3], 7);

        histogram = [1, 2, 3, 4];
        shift_histogram(&mut histogram, 2);
        assert_eq!(histogram[0], 0);
        assert_eq!(histogram[1], 0);
        assert_eq!(histogram[2], 1);
        assert_eq!(histogram[3], 9);

        histogram = [1, 2, 3, 4];
        shift_histogram(&mut histogram, 3);
        assert_eq!(histogram[0], 0);
        assert_eq!(histogram[1], 0);
        assert_eq!(histogram[2], 0);
        assert_eq!(histogram[3], 10);

        histogram = [1, 2, 3, 4];
        shift_histogram(&mut histogram, 4);
        assert_eq!(histogram[0], 0);
        assert_eq!(histogram[1], 0);
        assert_eq!(histogram[2], 0);
        assert_eq!(histogram[3], 10);

        // Shift left by varying amounts.
        histogram = [1, 2, 3, 4];
        shift_histogram(&mut histogram, -1);
        assert_eq!(histogram[0], 3);
        assert_eq!(histogram[1], 3);
        assert_eq!(histogram[2], 4);
        assert_eq!(histogram[3], 0);

        histogram = [1, 2, 3, 4];
        shift_histogram(&mut histogram, -2);
        assert_eq!(histogram[0], 6);
        assert_eq!(histogram[1], 4);
        assert_eq!(histogram[2], 0);
        assert_eq!(histogram[3], 0);

        histogram = [1, 2, 3, 4];
        shift_histogram(&mut histogram, -3);
        assert_eq!(histogram[0], 10);
        assert_eq!(histogram[1], 0);
        assert_eq!(histogram[2], 0);
        assert_eq!(histogram[3], 0);

        histogram = [1, 2, 3, 4];
        shift_histogram(&mut histogram, -4);
        assert_eq!(histogram[0], 10);
        assert_eq!(histogram[1], 0);
        assert_eq!(histogram[2], 0);
        assert_eq!(histogram[3], 0);
    }

}  // mod tests.
