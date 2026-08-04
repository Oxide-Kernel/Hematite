//! `DepthwiseConv2D` golden fixture — mirrors TFLM `reference_integer_ops::DepthwiseConvPerChannel`.

use crate::tflm_math;
use crate::fixture::FixtureWriter;

pub fn generate(w: &mut FixtureWriter) {
    // Shape: [1, 3, 3, 2] → batch=1, H=3, W=3, in_ch=2
    // Filter: [1, 3, 3, 4] → 1, H=3, W=3, out_ch=4 (depth_multiplier=2)
    // Output: [1, 3, 3, 4]
    let input_shape = [1i32, 3, 3, 2];
    let filter_shape = [1i32, 3, 3, 4];
    let output_shape = [1i32, 3, 3, 4];

    let input_height = 3i32;
    let input_width = 3i32;
    let input_depth = 2i32;
    let output_depth = 4i32;
    let filter_height = 3i32;
    let filter_width = 3i32;
    let depth_multiplier = 2;

    // Input: [1,3,3,2] = 18 values
    let input: Vec<i8> = (-9..9).map(|i| i as i8).collect();
    // Weights: [1,3,3,4] = 36 values
    let weights: Vec<i8> = (0..36).map(|i| ((i % 7) - 3) as i8).collect();
    let bias: Vec<i32> = vec![1, -1, 2, -2];

    let input_offset: i32 = 0;
    let output_offset: i32 = 0;
    let output_activation_min: i32 = -128;
    let output_activation_max: i32 = 127;

    let stride_width = 1;
    let stride_height = 1;
    // SAME padding with 3x3 kernel, stride 1
    let pad_width = 1;
    let pad_height = 1;
    let dilation_w = 1;
    let dilation_h = 1;

    // Per-channel output quantization
    let output_multiplier: Vec<i32> = vec![1i32 << 29, 1i32 << 30, 1i32 << 28, 1i32 << 29];
    let output_shift: Vec<i32> = vec![0, 0, 1, -1];

    let output_height = output_shape[1] as usize;
    let output_width = output_shape[2] as usize;
    let mut output: Vec<i8> = vec![0i8; output_height * output_width * output_depth as usize];

    for batch in 0..1i32 {
        for out_y in 0..(output_height as i32) {
            for out_x in 0..(output_width as i32) {
                for in_channel in 0..input_depth {
                    for m in 0..depth_multiplier {
                        let output_channel = m + in_channel * depth_multiplier;
                        let in_x_origin = (out_x * stride_width) - pad_width;
                        let in_y_origin = (out_y * stride_height) - pad_height;
                        let mut acc: i32 = 0;
                        for filter_y in 0..filter_height {
                            for filter_x in 0..filter_width {
                                let in_x = in_x_origin + dilation_w * filter_x;
                                let in_y = in_y_origin + dilation_h * filter_y;
                                let is_inside = in_x >= 0 && in_x < input_width
                                    && in_y >= 0 && in_y < input_height;
                                if is_inside {
                                    let in_idx = offset(&input_shape, batch, in_y, in_x, in_channel);
                                    let f_idx = offset(&filter_shape, 0, filter_y, filter_x, output_channel);
                                    let input_val = i32::from(input[in_idx]);
                                    let filter_val = i32::from(weights[f_idx]);
                                    acc += filter_val * (input_val + input_offset);
                                }
                            }
                        }
                        acc += bias[output_channel as usize];
                        let out_val = tflm_math::requantize_i8(
                            acc,
                            output_multiplier[output_channel as usize],
                            output_shift[output_channel as usize],
                            output_offset,
                            output_activation_min,
                            output_activation_max,
                        );
                        let out_idx = offset(&output_shape, batch, out_y, out_x, output_channel);
                        output[out_idx] = out_val;
                    }
                }
            }
        }
    }

    w.write("depthwise_conv2d",
        &input_shape, &filter_shape, &output_shape,
        &input, &weights, &bias,
        input_offset, output_offset,
        output_activation_min, output_activation_max,
        &output_multiplier, &output_shift,
        &output,
        &[("stride_width", stride_width), ("stride_height", stride_height),
          ("pad_width", pad_width), ("pad_height", pad_height),
          ("dilation_w", dilation_w), ("dilation_h", dilation_h),
          ("depth_multiplier", depth_multiplier)],
    );
}

fn offset(shape: &[i32; 4], d0: i32, d1: i32, d2: i32, d3: i32) -> usize {
    ((d0 as usize * shape[1] as usize + d1 as usize) * shape[2] as usize + d2 as usize)
        * shape[3] as usize + d3 as usize
}
