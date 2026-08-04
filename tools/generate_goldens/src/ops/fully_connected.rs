//! `FullyConnected` golden fixture — mirrors TFLM `reference_integer_ops::FullyConnected`.

use crate::tflm_math;
use crate::fixture::FixtureWriter;

pub fn generate(w: &mut FixtureWriter) {
    // Input: [1, 4] → batch=1, accum_depth=4
    // Filter: [3, 4] → output_depth=3, accum_depth=4
    // Bias: [3] → output_depth=3
    // Output: [1, 3]
    let input_shape = [1i32, 4, 1, 1]; // flat as NHWC
    let filter_shape = [3i32, 4, 1, 1]; // out_ch x accum_depth
    let output_shape = [1i32, 3, 1, 1];

    let accum_depth = 4usize;
    let output_depth = 3usize;
    let batches = 1usize;

    let input: Vec<i8> = vec![-5, 10, -3, 7];
    let weights: Vec<i8> = vec![
        1, 0, -1, 2,   // output channel 0
        -2, 1, 0, 1,    // output channel 1
        0, -1, 2, -1,   // output channel 2
    ];
    let bias: Vec<i32> = vec![0, 5, -5];

    let input_offset: i32 = 0;
    let output_offset: i32 = 0;
    let output_activation_min: i32 = -128;
    let output_activation_max: i32 = 127;

    // Per-channel output quantization (weights are symmetric, offset=0)
    let output_multiplier: Vec<i32> = vec![1i32 << 30, 1i32 << 29, 1i32 << 28];
    let output_shift: Vec<i32> = vec![0, 0, 1];

    let mut output: Vec<i8> = vec![0i8; batches * output_depth];

    for b in 0..batches {
        for out_c in 0..output_depth {
            let mut acc: i32 = 0;
            for d in 0..accum_depth {
                let input_val = i32::from(input[b * accum_depth + d]);
                let filter_val = i32::from(weights[out_c * accum_depth + d]);
                acc += filter_val * (input_val + input_offset);
            }
            acc += bias[out_c];
            let out_val = tflm_math::requantize_i8(
                acc,
                output_multiplier[out_c],
                output_shift[out_c],
                output_offset,
                output_activation_min,
                output_activation_max,
            );
            output[b * output_depth + out_c] = out_val;
        }
    }

    w.write("fully_connected",
        &input_shape, &filter_shape, &output_shape,
        &input, &weights, &bias,
        input_offset, output_offset,
        output_activation_min, output_activation_max,
        &output_multiplier, &output_shift,
        &output,
        &[("accum_depth", accum_depth as i32), ("output_depth", output_depth as i32)],
    );
}
