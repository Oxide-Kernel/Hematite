//! `Conv2D` golden fixture generator — mirrors TFLM `reference_integer_ops::ConvPerChannel`.
//!
//! Generates two fixtures: 1×1 conv and 3×3 conv, per the plan requirement.

use crate::tflm_math;
use crate::fixture::FixtureWriter;

/// Generate the conv2d 1×1 fixture (trivial input, acts like a fully-connected per-pixel).
pub fn generate_conv2d_1x1(w: &mut FixtureWriter) {
    // Shape: [1, 2, 2, 4] → batch=1, H=2, W=2, in_ch=4
    // Filter: [2, 1, 1, 4] → out_ch=2, H=1, W=1, in_ch=4 (1×1 kernel)
    // Output: [1, 2, 2, 2]
    let input_shape = [1i32, 2, 2, 4];
    let filter_shape = [2i32, 1, 1, 4];
    let output_shape = [1i32, 2, 2, 2];

    let batches = 1;
    let input_height = 2;
    let input_width = 2;
    let input_depth = 4;
    let output_depth = 2;
    let filter_height = 1;
    let filter_width = 1;

    // Synthetic input: small sequential int8 values
    let input: Vec<i8> = (-8..8).map(|i| i as i8).cycle().take(16).collect();
    // weights: per-channel
    let weights: Vec<i8> = vec![1, 2, 3, 4, -1, -2, -3, -4];
    // bias: int32 per output channel
    let bias: Vec<i32> = vec![10, -10];

    let input_offset: i32 = 0;
    let output_offset: i32 = 0;
    let output_activation_min: i32 = -128;
    let output_activation_max: i32 = 127;

    // Per-channel quantization: scale_out / (scale_in * scale_w[ch])
    let output_multiplier: Vec<i32> = vec![1i32 << 30, 1i32 << 28];
    let output_shift: Vec<i32> = vec![0, 1];

    let stride_width = 1;
    let stride_height = 1;
    let pad_width = 0;
    let pad_height = 0;
    let dilation_w = 1;
    let dilation_h = 1;

    let filter_input_depth = input_depth;
    let groups = input_depth / filter_input_depth;
    let filters_per_group = output_depth / groups;

    let output_height = output_shape[1] as usize;
    let output_width = output_shape[2] as usize;
    let mut output: Vec<i8> = vec![0i8; output_height * output_width * output_depth as usize];

    for batch in 0..batches {
        for out_y in 0..(output_height as i32) {
            let in_y_origin = (out_y * stride_height) - pad_height;
            for out_x in 0..(output_width as i32) {
                let in_x_origin = (out_x * stride_width) - pad_width;
                for out_channel in 0..output_depth {
                    let group = out_channel / filters_per_group;
                    let mut acc: i32 = 0;
                    for filter_y in 0..filter_height {
                        let in_y = in_y_origin + dilation_h * filter_y;
                        for filter_x in 0..filter_width {
                            let in_x = in_x_origin + dilation_w * filter_x;
                            let is_inside = in_x >= 0 && in_x < input_width
                                && in_y >= 0 && in_y < input_height;
                            if !is_inside {
                                continue;
                            }
                            for in_channel in 0..filter_input_depth {
                                let in_idx = offset_4d(
                                    &input_shape, batch, in_y, in_x,
                                    in_channel + group * filter_input_depth,
                                );
                                let f_idx = offset_4d(
                                    &filter_shape, out_channel, filter_y, filter_x, in_channel,
                                );
                                let input_val = i32::from(input[in_idx]);
                                let filter_val = i32::from(weights[f_idx]);
                                acc += filter_val * (input_val + input_offset);
                            }
                        }
                    }
                    acc += bias[out_channel as usize];
                    // Per-channel requantize + clamp
                    let out_val = tflm_math::requantize_i8(
                        acc,
                        output_multiplier[out_channel as usize],
                        output_shift[out_channel as usize],
                        output_offset,
                        output_activation_min,
                        output_activation_max,
                    );
                    let out_idx = offset_4d(
                        &output_shape, batch, out_y, out_x, out_channel,
                    );
                    output[out_idx] = out_val;
                }
            }
        }
    }

    w.write("conv2d_1x1",
        &input_shape, &filter_shape, &output_shape,
        &input, &weights, &bias,
        input_offset, output_offset,
        output_activation_min, output_activation_max,
        &output_multiplier, &output_shift,
        &output,
        &[("stride_width", stride_width), ("stride_height", stride_height),
          ("pad_width", pad_width), ("pad_height", pad_height),
          ("dilation_w", dilation_w), ("dilation_h", dilation_h)],
    );
}

/// Generate the conv2d 3×3 fixture (canonical edge-detection-like kernel).
pub fn generate_conv2d_3x3(w: &mut FixtureWriter) {
    // Shape: [1, 4, 4, 1] → single-channel 4×4 input
    // Filter: [1, 3, 3, 1] → 3×3 kernel, single input/out channel
    // Output with SAME padding (pads to keep spatial dims): [1, 4, 4, 1]
    let input_shape = [1i32, 4, 4, 1];
    let filter_shape = [1i32, 3, 3, 1];
    let output_shape = [1i32, 4, 4, 1];

    let batches = 1;
    let input_height = 4;
    let input_width = 4;
    let input_depth = 1;
    let output_depth = 1;
    let filter_height = 3;
    let filter_width = 3;

    // Diagonal ramp input
    let input: Vec<i8> = (0..16).map(|i| i as i8).collect();
    // Simple 3×3 kernel (identity-preserving-ish)
    let weights: Vec<i8> = vec![0, 0, 0, 0, 1, 0, 0, 0, 0];
    let bias: Vec<i32> = vec![0];

    let input_offset: i32 = 0;
    let output_offset: i32 = 0;
    let output_activation_min: i32 = -128;
    let output_activation_max: i32 = 127;

    let output_multiplier: Vec<i32> = vec![1i32 << 30];  // scale 0.5
    let output_shift: Vec<i32> = vec![0];

    let stride_width = 1;
    let stride_height = 1;
    // SAME padding: pad such that output spatial dims equal input spatial dims
    // For 3×3 kernel, stride 1: pad = 1 on each side
    let pad_width = 1;
    let pad_height = 1;
    let dilation_w = 1;
    let dilation_h = 1;

    let filter_input_depth = input_depth;
    let groups = input_depth / filter_input_depth;
    let filters_per_group = output_depth / groups;

    let output_height = output_shape[1] as usize;
    let output_width = output_shape[2] as usize;
    let mut output: Vec<i8> = vec![0i8; output_height * output_width * output_depth as usize];

    for batch in 0..batches {
        for out_y in 0..(output_height as i32) {
            let in_y_origin = (out_y * stride_height) - pad_height;
            for out_x in 0..(output_width as i32) {
                let in_x_origin = (out_x * stride_width) - pad_width;
                for out_channel in 0..output_depth {
                    let group = out_channel / filters_per_group;
                    let mut acc: i32 = 0;
                    for filter_y in 0..filter_height {
                        let in_y = in_y_origin + dilation_h * filter_y;
                        for filter_x in 0..filter_width {
                            let in_x = in_x_origin + dilation_w * filter_x;
                            let is_inside = in_x >= 0 && in_x < input_width
                                && in_y >= 0 && in_y < input_height;
                            if !is_inside {
                                continue;
                            }
                            for in_channel in 0..filter_input_depth {
                                let in_idx = offset_4d(
                                    &input_shape, batch, in_y, in_x,
                                    in_channel + group * filter_input_depth,
                                );
                                let f_idx = offset_4d(
                                    &filter_shape, out_channel, filter_y, filter_x, in_channel,
                                );
                                let input_val = i32::from(input[in_idx]);
                                let filter_val = i32::from(weights[f_idx]);
                                acc += filter_val * (input_val + input_offset);
                            }
                        }
                    }
                    acc += bias[out_channel as usize];
                    let out_val = tflm_math::requantize_i8(
                        acc,
                        output_multiplier[out_channel as usize],
                        output_shift[out_channel as usize],
                        output_offset,
                        output_activation_min,
                        output_activation_max,
                    );
                    let out_idx = offset_4d(
                        &output_shape, batch, out_y, out_x, out_channel,
                    );
                    output[out_idx] = out_val;
                }
            }
        }
    }

    w.write("conv2d_3x3",
        &input_shape, &filter_shape, &output_shape,
        &input, &weights, &bias,
        input_offset, output_offset,
        output_activation_min, output_activation_max,
        &output_multiplier, &output_shift,
        &output,
        &[("stride_width", stride_width), ("stride_height", stride_height),
          ("pad_width", pad_width), ("pad_height", pad_height),
          ("dilation_w", dilation_w), ("dilation_h", dilation_h)],
    );
}

/// 4D offset computation (NHWC layout, matching TFLM Offset).
fn offset_4d(shape: &[i32; 4], d0: i32, d1: i32, d2: i32, d3: i32) -> usize {
    ((d0 as usize * shape[1] as usize + d1 as usize) * shape[2] as usize + d2 as usize)
        * shape[3] as usize
        + d3 as usize
}
