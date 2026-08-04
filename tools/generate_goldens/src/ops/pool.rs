//! Pooling golden fixtures — mirrors TFLM `reference_integer_ops` `AveragePool` and `MaxPool`.

use crate::fixture::FixtureWriter;

pub fn generate_average_pool(w: &mut FixtureWriter) {
    // Input: [1, 4, 4, 1] → 4×4 single-channel
    // Output with 2×2 kernel, stride 2, VALID padding: [1, 2, 2, 1]
    let input_shape = [1i32, 4, 4, 1];
    let output_shape = [1i32, 2, 2, 1];

    // Input: ramp 0..15 scaled to interesting range
    let input: Vec<i8> = (-8..8).map(|i| i as i8).collect();

    let filter_width = 2;
    let filter_height = 2;
    let stride_width = 2;
    let stride_height = 2;
    let pad_width = 0;
    let pad_height = 0;
    let activation_min = -128i32;
    let activation_max = 127i32;

    let input_height = 4i32;
    let input_width = 4i32;
    let depth = 1i32;
    let output_height = 2usize;
    let output_width = 2usize;

    let mut output: Vec<i8> = vec![0i8; output_height * output_width];

    for out_y in 0..(output_height as i32) {
        for out_x in 0..(output_width as i32) {
            for ch in 0..depth {
                let in_x_origin = (out_x * stride_width) - pad_width;
                let in_y_origin = (out_y * stride_height) - pad_height;
                let fx_start = 0i32.max(-in_x_origin);
                let fx_end = filter_width.min(input_width - in_x_origin);
                let fy_start = 0i32.max(-in_y_origin);
                let fy_end = filter_height.min(input_height - in_y_origin);

                let mut acc: i32 = 0;
                let mut count: i32 = 0;
                for fy in fy_start..fy_end {
                    for fx in fx_start..fx_end {
                        let in_x = in_x_origin + fx;
                        let in_y = in_y_origin + fy;
                        let idx = (in_y * input_width + in_x) as usize * depth as usize + ch as usize;
                        acc += i32::from(input[idx]);
                        count += 1;
                    }
                }
                // Round to closest integer
                acc = if acc > 0 {
                    (acc + count / 2) / count
                } else {
                    (acc - count / 2) / count
                };
                acc = acc.max(activation_min).min(activation_max);
                let out_idx = (out_y as usize * output_width + out_x as usize) * depth as usize + ch as usize;
                output[out_idx] = acc as i8;
            }
        }
    }

    w.write("average_pool_2d",
        &input_shape, &[0; 4], &output_shape,
        &input, &[], &[],
        0, 0, activation_min, activation_max,
        &[], &[],
        &output,
        &[("filter_width", filter_width), ("filter_height", filter_height),
          ("stride_width", stride_width), ("stride_height", stride_height),
          ("pad_width", pad_width), ("pad_height", pad_height)],
    );
}

pub fn generate_max_pool(w: &mut FixtureWriter) {
    // Input: [1, 4, 4, 1] → 4×4 single-channel
    // Output with 2×2 kernel, stride 2, VALID: [1, 2, 2, 1]
    let input_shape = [1i32, 4, 4, 1];
    let output_shape = [1i32, 2, 2, 1];

    // Checkerboard pattern: alternating positive and negative
    let input: Vec<i8> = vec![1, -5, 3, -2, -8, 4, -1, 6, 7, -3, 0, -9, 2, -4, 5, -7];

    let filter_width = 2;
    let filter_height = 2;
    let stride_width = 2;
    let stride_height = 2;
    let pad_width = 0;
    let pad_height = 0;
    let activation_min = -128i32;
    let activation_max = 127i32;

    let input_height = 4i32;
    let input_width = 4i32;
    let depth = 1i32;
    let output_height = 2usize;
    let output_width = 2usize;

    let mut output: Vec<i8> = vec![0i8; output_height * output_width];

    for out_y in 0..(output_height as i32) {
        for out_x in 0..(output_width as i32) {
            for ch in 0..depth {
                let in_x_origin = (out_x * stride_width) - pad_width;
                let in_y_origin = (out_y * stride_height) - pad_height;
                let fx_start = 0i32.max(-in_x_origin);
                let fx_end = filter_width.min(input_width - in_x_origin);
                let fy_start = 0i32.max(-in_y_origin);
                let fy_end = filter_height.min(input_height - in_y_origin);

                let mut max_val = i8::MIN;
                for fy in fy_start..fy_end {
                    for fx in fx_start..fx_end {
                        let in_x = in_x_origin + fx;
                        let in_y = in_y_origin + fy;
                        let idx = (in_y * input_width + in_x) as usize * depth as usize + ch as usize;
                        max_val = max_val.max(input[idx]);
                    }
                }
                max_val = max_val.max(activation_min as i8).min(activation_max as i8);
                let out_idx = (out_y as usize * output_width + out_x as usize) * depth as usize + ch as usize;
                output[out_idx] = max_val;
            }
        }
    }

    w.write("max_pool_2d",
        &input_shape, &[0; 4], &output_shape,
        &input, &[], &[],
        0, 0, activation_min, activation_max,
        &[], &[],
        &output,
        &[("filter_width", filter_width), ("filter_height", filter_height),
          ("stride_width", stride_width), ("stride_height", stride_height),
          ("pad_width", pad_width), ("pad_height", pad_height)],
    );
}
